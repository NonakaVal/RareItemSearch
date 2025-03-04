# Coleta e Análise de Preços de Produtos com Agentes Automáticos

Este projeto utiliza a plataforma **CrewAI** para criar agentes automatizados que realizam a coleta, análise e geração de relatórios sobre preços e disponibilidade de produtos em plataformas de e-commerce populares como **Amazon**, **Mercado Livre** e **eBay**. O sistema facilita a comparação de preços, frete e estoque de diferentes plataformas e gera recomendações de compra.

## Funcionalidades

O sistema é composto por uma série de agentes que desempenham as seguintes funções:

1. **Coleta de Dados de Preços e Disponibilidade**:
   - O agente coleta dados sobre o preço do produto, a quantidade de estoque disponível, as opções de frete e os links para as plataformas de e-commerce pesquisadas.
   - As plataformas de e-commerce pesquisadas incluem **Amazon**, **Mercado Livre** e **eBay**.

2. **Análise de Dados**:
   - O agente compara os dados coletados de diferentes plataformas de e-commerce, levando em consideração o preço, a quantidade de estoque disponível, as opções de frete e a confiabilidade dos links.
   - Ele identifica as melhores ofertas com base no custo-benefício.

3. **Revisão de Dados**:
   - O agente verifica a precisão e consistência dos dados coletados, garantindo que as informações estejam corretas antes de serem incluídas no relatório final.
   - A revisão inclui a verificação de preços, estoque, opções de frete e a validação dos links dos produtos.

4. **Redação de Relatório**:
   - O agente redige um relatório detalhado sobre o produto, incluindo uma análise comparativa de preços e recomendações de compra.
   - O relatório final inclui links para os produtos nas plataformas pesquisadas, facilitando o acesso rápido às ofertas.

## Pré-requisitos

- **Python 3.8+**
- **Bibliotecas**:
  - `crewai`: Biblioteca para gerenciamento de agentes e processos.
  - `crewai_tools`: Ferramentas de raspagem e pesquisa.
  - `langchain_openai`: Biblioteca para integração com a API do OpenAI.
  - `dotenv`: Para carregar variáveis de ambiente.

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/NonakaVal/RareItemSearch.git
cd RareItemSearch
```

### Crie e ative o ambiente virtual:

```bash
virtualenv venv
source venv/bin/activate  # Para sistemas Unix ou MacOS
venv\Scripts\activate  # Para sistemas Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Configure as variáveis de ambiente

Você precisará das chaves de API do **OpenAI** e **Serper** para o funcionamento do código. Crie um arquivo `.env` na raiz do projeto e adicione as seguintes variáveis:

```ini
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

### 4. Verifique e instale o arquivo `requirements.txt`

Se o arquivo `requirements.txt` não estiver presente ou se você preferir instalar as dependências manualmente, use o seguinte comando para instalar as bibliotecas necessárias:

```bash
pip install crewai langchain_openai crewai_tools dotenv
```

## Como Usar

1. Execute o script principal para iniciar o processo de coleta de dados, análise e geração do relatório.

```bash
python main.py
```

2. O script solicitará que você insira as seguintes informações:

   - **Nome do produto** (ex: Playstation 5, Xbox Series X, etc.).
   - **Categoria do produto** (ex: videogame, acessório, etc.).
   - **Marca do produto** (ex: Sony, Microsoft, Logitech).
   - **Edição do produto** (se aplicável, ex: edição limitada, bundle, etc.).

3. O script solicitará também o **diretório de saída** onde os resultados serão salvos. Se o diretório não existir, ele será criado automaticamente.

4. O processo será executado sequencialmente, passando por todas as etapas de coleta de dados, análise, revisão e redação do relatório. O relatório final será salvo no diretório de saída especificado.

## Estrutura do Código

O código é dividido em várias seções:

1. **Configuração de APIs**:
   - Carrega as chaves de API do OpenAI e Serper através do arquivo `.env`.

2. **Inicialização de Ferramentas**:
   - Ferramentas de raspagem e pesquisa para buscar dados sobre os produtos em plataformas de e-commerce.

3. **Definição de Agentes**:
   - Quatro agentes são definidos para realizar diferentes tarefas: coleta de dados, análise de dados, revisão de dados e redação de relatório.

4. **Definição de Tarefas**:
   - As tarefas são atribuídas aos agentes e incluem a coleta de dados, análise comparativa, revisão de dados e geração do relatório final.

5. **Execução do Processo**:
   - A equipe de agentes é gerenciada e as tarefas são executadas sequencialmente, com o resultado final sendo impresso no terminal.

## Saída

Os resultados das tarefas são salvos em arquivos Markdown no diretório especificado:

- `1-dados-precos.md`: Dados coletados sobre o produto (preços, estoque, frete e links).
- `2-analise-precos.md`: Relatório de análise comparativa das ofertas.
- `3-revisao-precos.md`: Relatório de revisão da precisão dos dados coletados.
- `4-relatorio-final.md`: Relatório final com as recomendações de compra.

## Contribuições

Contribuições para melhorar o sistema são bem-vindas! Para contribuir, siga estas etapas:

1. Faça um fork do repositório.
2. Crie uma branch para a sua feature (git checkout -b feature/nome-da-feature).
3. Faça suas alterações e commit (git commit -m 'Adiciona nova feature').
4. Envie para o repositório remoto (git push origin feature/nome-da-feature).
5. Abra um pull request para a branch principal.

## Licença

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
```
