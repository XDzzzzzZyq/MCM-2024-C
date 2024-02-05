### MCM-2024
MCM-2024 Problem C

# Psychological Angular Momentum Model (PAMM)
![54e8c2e506de8d0d6cf3ed72609df68](https://github.com/XDzzzzzZyq/MCM-2024-C/assets/81028185/c074f48c-13be-4a8c-94ce-de747bed51bb)

The probability of a player winning in the game is not entirely determined by the intrinsic
ability of the player, otherwise, it would be hard to explain why the results of the game between
two players who sometimes seem to have a disparity in strength will be unexpected, and why the
results often reverse in the game.

We introduce the **Psychological Angular Momentum Model (PAMM)**   to address the problem.
In the model, we make an analogy between the competition of two players and the interaction
between two spinning tops. We quantify players’ psychological momentum by the actual angular
momentum of the **spinning tops**, and their intrinsic abilities by their moment of inertia.

- For task 1, our core aim is to visualize and explain which player has a better performance and
how the flow changes. We first construct the momentum curve for both players by calculating
**moment increments** and summing them up. Then we use** rotational kinetic energy** to represent
players’ possibility of winning. By calculating the **Performance Coefficient** we get to know who
is at better performance, which is correlated to match data.

- For task 2, our goal is to analyze **the Random hypothesis of PAM**. After calculating the **Pearson’s Correlation Coefficient**, we reject this hypothesis by visualizing the Correlation Matrix
The result turns out that the momentum is not random, instead, is highly related to match features.

- For task 3, By adding or subtracting variables in the model and observing the changes in
momentum fluctuations, we obtain the degree of correlation between different variables and momentum. In order to find the indicator of the swing change during games, we compare the points
where influential parameters occur and the parameter curve with the momentum curve of the player
to see if the density of points or changes in parameter curve will affect the the **flow of game**, so as
find that in general, **serving speed** and number of **unforced errors** are most related to momentum
flow and can be used as indicators of the swings'

- For task 4, we migrate the pre-trained parameters from other men’s tennis matches and find
that our model is still effective, regardless of slight differences in different players’ **sensitivity** to
different parameters. To examine the **ability of generalization**, we first process the dataset of 2011
ITF Women’s World Tennis Tour, sort out the available parameters (missing four out of nine
parameters we used before), and bring in the data. The result shows that the sum of the two players’
momentum is nearly zero. The differences in results suggest that the missing parameters are also
essential.

**Keywords: Psychological momentum; Angular momentum; Linear regression.**
