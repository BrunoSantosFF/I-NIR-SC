# Fusão de Imagens e Dados Espectrais NIR para Classificação de Câncer de Pele

Este repositório reúne o código utilizado no artigo [Colocar artigo], disponibilizando scripts e ferramentas voltados à análise e classificação de lesões de pele por meio de modelos de aprendizado profundo.

## Funcionalidades

- Implementações de redes neurais convolucionais (CNNs) para análise de imagens de câncer de pele
- Classificação e treinamento de uma rede neural MLP
- Cálculo e visualização de valores SHAP


## Dependências
O código é desenvolvido em Python e utiliza as seguintes bibliotecas principais: 

- PyTorch: Para a implementação dos modelos de aprendizado profundo.
- Scikit-Learn: Para utilidades complementares de aprendizado de máquina.


## Estrutura do Repositório

- my_models: Contém as implementações dos modelos CNN.

- benchmarks: Inclui scripts para executar experimentos em cada conjunto de dados.

- dataset: Contém as pastas dos conjuntos de dados utilizados.


##  Links para os Conjuntos de Dados
Os conjuntos de dados utilizados neste projeto estão disponíveis publicamente para download no Data Mendeley:

PAD-UFES-20: Conjunto de dados de imagens de lesões de pele, incluindo metadados, para diagnóstico automatizado.

FLUO-SC: Conjunto de dados contendo imagens de fluorescência de lesões de pele.

Consulte as respectivas páginas dos conjuntos de dados para mais informações sobre seu conteúdo e uso.


🛠 Instruções de Uso
Passo 1: Clonar o Repositório

Passo 2: Configurar os Caminhos dos Conjuntos de Dados
Para cada benchmark, você precisa definir o caminho do conjunto de dados nos scripts correspondentes dentro da pasta benchmarks.

Passo 3: Executar Experimentos

Dicas e Recomendações
Gerenciamento de Experimentos com Sacred: Sacred é uma ferramenta robusta para organizar experimentos, rastrear configurações e salvar resultados. Se você não estiver familiarizado, consulte a documentação do Sacred.

Modificações Personalizadas: Você pode modificar os scripts na pasta benchmarks para adaptá-los a conjuntos de dados ou configurações experimentais específicas.

Reprodutibilidade: Certifique-se de que os caminhos dos conjuntos de dados e as configurações do sistema estão corretamente definidos para reproduzir os resultados.

Créditos e Agradecimentos
Parte dos códigos neste repositório são adaptações do repositório de André Pacheco, disponível em https://github.com/paaatcha/my-thesis.

Essas adaptações foram essenciais para:

Garantir a reprodutibilidade dos resultados obtidos anteriormente com o conjunto de dados PAD-UFES-20.

Modificar e estender as implementações originais para incorporar a abordagem de aprendizado por transferência.

Somos gratos pela base fornecida pelo trabalho de André Pacheco, que contribuiu significativamente para o sucesso deste projeto.

https://chat.deepseek.com/share/846s2wexukd75nmeq2