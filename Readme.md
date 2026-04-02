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

Para configurar o ambiente, certifique-se de que o Python está instalado no seu sistema. Em seguida, instale as dependências necessárias usando:

```
pip install -r requirements.txt
```


## Estrutura do Repositório

- class: Contém as implementações dos modelos CNN e scripts para executar experimentos.

- shap: Script da estrtura do SHAP 

- inputs: Contém as pastas dos conjuntos de dados utilizados.


##  Links para os Conjuntos de Dados
Os conjuntos de dados utilizados neste projeto estão disponíveis publicamente:

--- colocar o dataset [I+NIR-SC]


## Instruções de Uso
Passo 1: Clonar o Repositório

```
git clone https://github.com/BrunoSantosFF/I-NIR-SC.git
```

Passo 2: Configurar os Caminhos dos Conjuntos de Dados

Dentro do arquivo principal (**main.py**), na varivel `MODEL_CONFIGS` configure os arquivos de entrada para cada modelo escolhido.

Passo 3: Executar Experimentos

Para executar o experimento, basta usar o script `main.py` com o seguinte comando e ira aparecer um menu interativo: 

```
python3 main.py 
```

## Contato
Para qualquer dúvida, entre em contato comigo em brunosf1804@gmail.com

## Créditos e Agradecimentos
Algumas implementações dos códigos foi baseada nestes repositórios, disponíveis em https://github.com/paaatcha/my-thesis e https://github.com/MatheusBecali/VAE-GNA.git.

Essas adaptações foram essenciais para:

- Garantir a reprodutibilidade dos resultados obtidos anteriormente com o conjunto de dados PAD-UFES-20 e ser retreinado (Fine-tuning) com o dataset do I+NIR-SC e salvar suas features para ser utilizadas do codigo.

- A mesma coisa foi executada no conjunto de dados NIR-SC-UFES e foram obtidas o vetor latente para os inputs.



