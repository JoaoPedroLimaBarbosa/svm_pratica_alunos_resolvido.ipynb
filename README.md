# SVM — o Titanic contra a Random Forest

Prática de IA sobre **Máquinas de Vetores de Suporte (SVM)**. O modelo é usado para prever a sobrevivência dos passageiros do Titanic e comparado com uma Random Forest de referência.

## Objetivo

Entender como o SVM funciona e como seus hiperparâmetros afetam o resultado: a **escala dos dados**, o parâmetro **C**, o **kernel** e o **gamma**. No final, comparar o melhor SVM com a Random Forest usando treino × validação.

## Dados

Dataset **Titanic** (Kaggle / seaborn-data): 891 passageiros, dos quais **38% sobreviveram**.

Variáveis usadas: `pclass` (classe do bilhete), `sex`, `age`, `sibsp` e `parch` (familiares a bordo), `fare` (valor da passagem) e `embarked` (porto de embarque).

Tratamentos aplicados:
- idades vazias preenchidas com a mediana;
- portos de embarque vazios preenchidos com `"S"`;
- `sex` convertido para 0/1 e `embarked` transformado em colunas 0/1 (one-hot encoding).

## Metodologia

- Validação cruzada estratificada com 5 partes (`StratifiedKFold`, `random_state=42`);
- Métrica: acurácia no treino e na validação;
- Modelos: `RandomForestClassifier` (300 árvores, profundidade máxima 6) e `SVC`;
- Escala com `StandardScaler` dentro de um `make_pipeline`;
- Busca automática de hiperparâmetros com `GridSearchCV` (C em `[0.1, 1, 10]` e gamma em `[0.01, 0.1, 1]`).

## Resultados

| Modelo | Treino | Validação |
|---|---|---|
| Random Forest (referência) | 87,3% | **83,3%** |
| SVM sem escala | 68,3% | 67,8% |
| SVM com escala | 84,1% | 82,7% |
| SVM `C=0,01` | 61,6% | 61,6% |
| SVM `C=100` | 88,0% | 79,8% |
| SVM kernel linear | 79,0% | 78,7% |
| SVM `gamma=10` | 91,2% | 75,0% |

Melhor combinação do `GridSearchCV`: `C=1` e `gamma=0,1`, com 82,5% de validação.

Os valores podem variar um pouco entre execuções e ambientes.

## Principais conclusões

- **A escala é essencial no SVM:** sem `StandardScaler`, a validação foi de 67,8%; com escala, 82,7%. O SVM decide por distância, e a coluna `fare` (de 0 a 512) dominava as demais.
- **C muito baixo (0,01)** deixa o modelo simples demais: 61,6% de validação, que é exatamente a proporção de passageiros que morreram, ou seja, o modelo prevê "morreu" para todo mundo.
- **C muito alto (100)** causa overfitting: treino de 88,0% e validação de 79,8%.
- **Kernel rbf foi melhor que o linear** (82,7% contra 78,7%), o que sugere que a fronteira entre sobreviventes e não sobreviventes não é uma reta.
- **Gamma alto (10)** causa overfitting forte: treino de 91,2% e validação de 75,0%.
- **A Random Forest ganhou por pouco** (83,3% contra 82,5% do melhor SVM da busca). Não existe modelo campeão universal, por isso é preciso testar.

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/SEU_USUARIO/svm-titanic.git
   cd svm-titanic
   ```
2. Instale as dependências:
   ```bash
   pip install pandas scikit-learn jupyter
   ```
3. Abra o notebook `svm_pratica_alunos_resolvido.ipynb` e execute todas as células. O dataset é baixado automaticamente da URL indicada no código, então é preciso ter conexão com a internet.

## Tecnologias

Python, pandas, scikit-learn e Jupyter Notebook.

## Autor

Joao Pedro Lima Barbosa 
