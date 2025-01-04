from builtins import input, print
import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai_tools import SerplyWebSearchTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurações do OpenAI e Serper (APIs externas)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Inicialização do modelo de linguagem (LLM)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=1500,    
)

# Inicialização das ferramentas de pesquisa e raspagem de dados
search_tool = SerperDevTool(n_results=12)
scrape_tool = ScrapeWebsiteTool()

tools = [search_tool, scrape_tool]

# Função para selecionar diretório de saída para os resultados
def select_output_directory():
    output_dir = input("Escolha o nome do diretório para salvar os resultados. O diretório será criado se não existir: \n")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

# Solicitação de entradas do usuário
output_directory = select_output_directory()

# Coleta de informações do produto
product_name = input("Digite o nome completo do produto que deseja pesquisar (ex: Playstation 5, Xbox Series X, etc.): \n")
product_category = input("Digite a categoria do produto (ex: videogame, acessório, etc.): \n")
product_brand = input("Digite a marca do produto (ex: Sony, Microsoft, Logitech): \n")
product_edition = input("Digite a edição do produto, se aplicável (ex: edição limitada, bundle, etc.): \n")

############################################################################################################
# Definição de agentes para execução de tarefas
############################################################################################################

# Agente de Coleta de Dados de Preços
data_collector = Agent(
    role="Coletor de Dados de Preços e Disponibilidade",
    goal=f"Pesquisar o produto '{product_name}' nas plataformas online mais populares (como Amazon, Mercado Livre, eBay) para coletar informações detalhadas sobre preços, disponibilidade, fretes e links dos produtos.",
    backstory="Este agente é responsável por realizar a coleta de dados de preços, estoque, condições de frete e links de produtos de plataformas de e-commerce confiáveis. Ele deve buscar informações sobre o preço atual do produto, a quantidade em estoque, a disponibilidade de frete e coletar os links diretos para os produtos nas plataformas pesquisadas.",
    llm=llm,
    allow_delegation=False,
    tools=tools
)

# Agente de Análise de Dados
data_analyzer = Agent(
    role="Analista de Dados de Mercado",
    goal=f"Analisar as informações coletadas sobre o produto '{product_name}' para identificar as melhores ofertas, levando em conta o preço, a disponibilidade, o frete e os links dos produtos.",
    backstory="Este agente irá comparar os dados coletados das diferentes plataformas de e-commerce para fornecer uma análise comparativa detalhada, identificando as melhores opções de compra com base em custo-benefício e considerando os links para cada produto.",
    llm=llm,
    allow_delegation=False,
    tools=tools
)

# Agente de Revisão de Dados
data_reviewer = Agent(
    role="Revisor de Dados de Preços",
    goal=f"Revisar os dados de preços e disponibilidade do produto '{product_name}' para garantir que todas as informações estão corretas, consistentes e precisas, incluindo os links dos produtos.",
    backstory="Este agente verifica a precisão e integridade dos dados de preços, disponibilidade, frete e links coletados, realizando uma checagem de qualidade antes de gerar o relatório final.",
    llm=llm,
    allow_delegation=False,
    tools=tools
)

# Agente de Escrita de Relatório
report_writer = Agent(
    role="Escritor de Relatório de Preços e Disponibilidade",
    goal=f"Redigir um relatório detalhado e bem estruturado sobre o produto '{product_name}', incluindo uma análise de preços, disponibilidade, links dos produtos e recomendações sobre as melhores opções de compra.",
    backstory="Este agente é responsável por compilar as informações coletadas e analisadas em um relatório claro e informativo, incluindo os links dos produtos nas plataformas e oferecendo insights valiosos sobre as melhores ofertas.",
    llm=llm,
    allow_delegation=False,
    tools=tools
)

############################################################################################################
# Definição das tarefas para os agentes
############################################################################################################

# Tarefa de Coleta de Dados de Preços
data_collection_task = Task(
    description=f"""
    Pesquisar o produto '{product_name}' nas plataformas de e-commerce como Amazon, Mercado Livre e eBay. 
    Coletar dados detalhados sobre o preço do produto, a quantidade de estoque disponível, as opções de frete oferecidas pelas plataformas e os links diretos dos produtos.
    O objetivo é obter informações completas e precisas sobre o produto em cada plataforma para uma comparação eficaz.
    """,
    expected_output="Documento ou planilha contendo informações detalhadas sobre os preços, a disponibilidade de estoque, as opções de frete e os links dos produtos nas plataformas pesquisadas.",
    agent=data_collector,
    output_file=os.path.join(output_directory, "1-dados-precos.md")
)

# Tarefa de Análise de Dados
data_analysis_task = Task(
    description=f"""
    Analisar as informações coletadas sobre o produto '{product_name}', comparando os preços, a disponibilidade de estoque, as opções de frete e os links diretos dos produtos de diferentes plataformas. 
    Identificar as melhores ofertas com base no custo-benefício, levando em consideração preço, estoque, frete e os links para acesso rápido às ofertas.
    """,
    expected_output="Relatório de análise comparativa destacando as melhores ofertas, incluindo links diretos para os produtos e as plataformas mais vantajosas em termos de preço, disponibilidade e frete.",
    agent=data_analyzer,
    context=[data_collection_task],
    output_file=os.path.join(output_directory, "2-analise-precos.md")
)

# Tarefa de Revisão de Dados
data_review_task = Task(
    description=f"""
    Revisar as informações de preços, disponibilidade, frete e links coletados sobre o produto '{product_name}', garantindo a precisão dos dados e a consistência das informações entre as fontes.
    A revisão deve incluir a verificação de valores de preço e estoque, confirmar que as informações de frete são precisas e validar os links fornecidos para garantir que os produtos estão acessíveis.
    """,
    expected_output="Relatório com feedback detalhado sobre a precisão e qualidade dos dados coletados, incluindo uma verificação completa dos links dos produtos.",
    agent=data_reviewer,
    context=[data_analysis_task],
    output_file=os.path.join(output_directory, "3-revisao-precos.md")
)

# Tarefa de Redação de Relatório Final
final_report_task = Task(
    description=f"Redigir um relatório completo e bem estruturado sobre o produto '{product_name}', incluindo uma análise de preços, disponibilidade, links dos produtos e recomendações de compra.",
    expected_output="Relatório final detalhado que resuma todas as informações coletadas e analisadas, apresentando uma comparação das melhores opções de compra e destacando os melhores preços, condições de disponibilidade e links para acesso rápido às ofertas.",
    agent=report_writer,
    context=[data_review_task],
    output_file=os.path.join(output_directory, "4-relatorio-final.md")
)

############################################################################################################
# Definição da equipe de agentes (Crew)
############################################################################################################

# Configuração da equipe (Crew)
crew = Crew(
    agents=[data_collector, data_analyzer, data_reviewer, report_writer],  # Lista de agentes
    tasks=[data_collection_task, data_analysis_task, data_review_task, final_report_task],  # Tarefas para execução
    process=Process.sequential,  # Execução das tarefas de forma sequencial
    manager_llm=llm,  # LLM para gerenciar os agentes
    verbose=True  # Ativar saída detalhada
)

# Iniciando o processo com a execução das tarefas
result = crew.kickoff()
print(result)
