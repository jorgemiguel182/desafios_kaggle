library(dplyr)
library(readr)
detach(package:tidyr)

# Prevendo os sobreviventes do titanic

# Importando dados
?read.csv
train <- read.csv('train.csv')
test <- read.csv('test.csv')

train$tp <- 'train'
test$tp <- 'test'
test$Survived <- NA

summary(train)
summary(test)

# Criando dataset com todos os dataframes
?bind_rows
full <- bind_rows(train, test)

# Analisando dados
str(full)
summary(full)


# Verificando idades nul
hist(full$Age)

# Testando pacote mice para imputacao de idades missing
install.packages('mice')
library(mice)

md.pattern(full)

install.packages('VIM')
library(VIM)

aggr_plot <- aggr(full, col=c('navyblue','red'), numbers = TRUE, sortVars = TRUE, labels=names(full), cex.axis=.7, gap=3, ylab=c("Histogram of missing data","Pattern"))

tempData <- mice(full, m=5, maxit = 50, meth = 'pmm', seed = 500)

summary(tempData)

tempData$meth

completeData <- complete(tempData)

# Plot age distributions
par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(completeData$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Substituindo as idades originais pelas previstas
full$Age <- completeData$Age

sum(is.na(full$Age))

str(full)
summary(full)


head(full)

full$fSize <- full$SibSp + full$Parch + 1
full$child[full$Age >= 18] <- 'Adulto'
full$child[full$Age < 18] <- 'Criança'

full$child <- factor(full$child)

full_1 <- full

sum(full$Cabin == '')

full_1$Cabin[is.na(full_1$Cabin)] <- "U"
full_1$Cabin <- substring(full_1$Cabin, 1, 1)
full_1$Cabin <- as.factor(full_1$Cabin)

sum(is.na(full$Cabin))

head(full$Cabin, 50)

str(full_1)

# Fazendo o split dos dados
train_limpo <- filter(full_1, tp == 'train')
test_limpo <- filter(full_1, tp == 'test')

dados_treino <- train_limpo
dados_treino <- filter(dados_treino, tp == 'train')
dados_treino <- select(dados_treino, -Name, -Ticket, -Cabin, -PassengerId, -tp, -Embarked)
dados_treino$Survived <- as.character(dados_treino$Survived)
dados_treino$Survived <- as.factor(dados_treino$Survived)


dados_teste <- test_limpo
dados_teste <- filter(dados_teste, tp == 'test')
dados_teste <- select(dados_teste, -Name, -Ticket, -Cabin, -PassengerId, -tp, -Embarked)

str(dados_treino)
summary(dados_treino)

sum(is.na(dados_treino))




# Treinando modelo RandomForest
library(randomForest)
model.rf <- randomForest(Survived ~ . , data = dados_treino, importance = TRUE)

summary(model.rf)






# Treinando o modelo
log.model <- randomForest(Survived ~ . , data = dados_treino, importance = TRUE)

# Podemos ver que as variaveis Sex, Age e Pclass sao as variaveis mais significantes
summary(log.model)

# Fazendo as previsoes nos dados de teste
library(caTools)
set.seed(101)

# Split dos dados
split = sample.split(dados_treino$Survived, SplitRatio = 0.70)

# Datasets de treino e de teste
dados_treino_final = subset(dados_treino, split == TRUE)
dados_teste_final = subset(dados_treino, split == FALSE)

# Gerando o modelo com a versao final do dataset
final.log.model <- glm(formula = Survived ~ . , family = binomial(link='logit'), data = dados_treino_final)

# Resumo
summary(final.log.model)

# Prevendo a acuracia
fitted.probabilities <- predict(final.log.model, newdata = dados_teste_final, type = 'response')
head(fitted.probabilities)

# Calculando os valores
fitted.results <- ifelse(fitted.probabilities > 0.5, 1, 0)

# Conseguimos quase 80% de acuracia
misClasificError <- mean(fitted.results != dados_teste_final$Survived)
print(paste('Acuracia', 1-misClasificError))

# Criando a confusion matrix
table(dados_teste_final$Survived, fitted.probabilities > 0.5)

prediction <- predict(final.log.model,dados_teste )

solution <- data.frame(PassengerID = test_limpo$PassengerId, Survived = prediction)

write.csv(solution, file = 'solution.csv', row.names = F)

head(solution, 70)





