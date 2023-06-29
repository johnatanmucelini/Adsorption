# Começo

- Salve trabalhadores da ciência do Brasil e do mundo!
- Meu nome é Johnatan
- Objetivo. Nesse vídeo eu vou apresentar um algoritmo que eu desenvolvi com minha amiga Lucas pro doutorado dela.

# Resumo

A ideia do algorithmo: é vc fornece duas estrutruas atômicas, por exemplo, A e B, e ele cria várias estruturas AB adsorvidas que são importantes pra estudo de interações intermoleculares, tanto, através de simulações via mecânica quântica quanto mecânica classica. Ele ainda pode ser utilizado pra outros propósitos como eu vou explicar no final do vídeo.

Vc escolher quantas estruturas AB vão ser criadas, mas independente do número de estruturas, elas são criadas para serem diferentes entre si. Isso é muito importante, tanto pra explorar o máximo do espaço das estruturas AB, quanto pra evitar simular estruturas muito parecidas ou iguais por simetria, já que elas vão te levar a resultados muito parecidos ou iguais.

Na prática, vc da duas moléculas A e B no formato xyz e alguns parâmetros, e o algoritimo devolve um grupo de estruturas ABs com o tamanho que vc escolhe.

Bom, essa é a ideia geral. Agora vamo entrar na metodologia.

# Metodologia

Eu divido ela em 3 partes.

* A primeira parte é o mapeamento das superfícies de A e B.
  * O objetivo dessa parte é encontrar um grupo de pontos na superfície de cada molécula que representem diferentes ambientes químicos em cada molécula.
* A segunda é a adsorção de A e B
  * Aqui o objetivo é combinar A e B através dos diferentes ambientes químicos de cada molécula e gerar um grupo grande de estruturas
* E a terceira é extração de um set representativo de estruturas AB
  * O objetivo aqui é representar esse grupo grande de estruturas através de um grupo menor de tamanho a ser escolhido pelo usuário

Vamo pra primeira parte da metodologia.

## Mapeamento

Ele inicia com a definição de raios de van der Walls. Já existem raios pra alguns átomos no código, mas vc pode adicionar outros ou modificar os que tão la, só olhar o arquivo adsorption.py.

Com eles vamos mapear uma superfície da molécula baseado em átomos como esferas rígidas. O raio dessas esferas é o raio de van der Walls. Como resultado nos pegamos uma grupo de pontos formando a superfície da molécula, tanto de A quanto de B. Pra encontrar as posições de cada ponto é um procedimento um pouco que não vou entrar em detalhes aqui, mas tem detalhes e referências no readme. Aqui os pontos dessa superfície tão coloridos por átomo.

Agora, encontrar os ambientes químicos. Explicando de uma forma mais lúdica, vamos investigar o ambiente químico perto de cada ponto dessa superfície. Ai vamos agrupar os pontos de ambientes químicos parecidos. E pra cada ambiente químico vamos pegar o ponto que é mais parecido com média dos ambientes químicos dos pontos do grupo. Na pratica isso é um processo de calcular descritores e fazer um uma análise de clustering. Não vou me aprofundar na metodologia, mas tem mais informações no readme. O importante desse passo é que vc precisa definir quantos ambientes químicos devem ser procurados em cada uma das moléculas A e B. (Isso é um parâmetro do algorítmo que quanto maior, melhor tende a ser o resultado final do algorítmo, mas também mais demorado ele vai ser, isso vai ficar mais claro depois). Aqui os pontos dessa superfície estão colorizados por ambiente químico e o pontos maiores são os pontos representativos desse ambiente químico. Note como esse procedimento percebe a simetria local das moléculas. Pra visualizar o resultado dessa análise é feita a redução de dimensionalidade dos features, com t-SNE, e apresentado um mapa 2d dos clusters encontrados.

Agora vamos pra segunda parte da metodologia

## Adsorção

Agora vamos combinar as moléculas A e B através de seus ambientes químicos. Todos os ambientes químicos de A contra todos os ambientes químicos de B. A combinação direta deles pode sobrepor as duas moléculas. E mesmo se não sobrepor, existem várias forma de encaixar A e B através de um mesmo ambientes. Então, vamos gerar ABs rotacionando a molécula B de várias formas antes da combinação. Essas rotações são feitas através de um grid no espaço das possíveis rotações que não vou entrar em detalhes, mas tem mais informações no readme. Na prática, o q é importante sobre as rotações é o número total de rotações desejado que o algoritmo faz o resto. O número final de rotações pode mudar levemente. Mais informações no readme.

Desse modo, o numero total de estruturas testadas é igual ao produto do número de ambientes químicos em A, em B e o número de rotações.

As moléculas AB geradas assim são filtradas e vão pra um poll de estruturas. O filtro é que a molécula não se sobreponham nem sejam similares a alguma estrutura já adicionada no poll. AB são consideradas sobrepostas qualquer par de átomos estiver a uma distância menor que soma de seus raios de van der Walls multiplicado por uma parâmetro que vc escolhe. Duas estruturas são consideradas similares se a distância entre elas num espaço de descritores for menor que um parâmetro que vc escolhe. Esse descritores são uma concatenação dos descritores análogos aos utilizados pra detectar os ambientes químicos dos pontos da superfície da molécula, mais detalhes no readme.

No final do processo o pool deve ter muito mais estruturas que o número de estruturas desejadas. O último passo é da metodologia é a seleção de um grupo representativo.

## Seleção de um grupo representativo

Agora vamos selecionar um grupo representativo do pool de estruturas AB. Para isso, utilizaremos os mesmos descritores do passo anterior para fazer uma análise de clustering, que classifica os ABs em grupos de ABs parecidos. Para cada grupo de ABs, selecionaremos a estrutura com características mais próximas a média do grupo. A quantidade de grupos é definida pelo usuário. O resultado desse clustering é mostrado num mapa 2d através de uma redução de dimensionalidade com t-SNE.
