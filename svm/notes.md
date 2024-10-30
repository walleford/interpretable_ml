## Support Vector Classifiers

- Idea is to set the threshold  for classification between the edge cases of each class

- the shortest distance between observations and the threshold is called the margin

- when the threshold is halways between the two edge observations, the margin is as large as it can be

- When you use the threshold that gives the largest margin to make classifications, 
you are using a Maximal Margin Classifier

- to make a threshold that isn't sensitive to outliers, we must allow misclassifications

- choosing a threshold that allows misclassifications is an example of the bias/variance tradeoff that
plagues all machine learning

- a threshold with higher bias with have lower variance

- when we allow misclassifications, the distance between the observations and the threshold is called
a soft margin

- use cross validation to determine how many misclassifications and observations to allow inside the soft margin
to get the best classification

- when the data is 2-dimensional, the support vector classifier is a line

- if the data is 3-dimensional, the support vector classifier is a plane

- when the data is 4-dimensional or more, the support vector classifier is a hyperplane

### Support Vector Machines

- start by taking a 1-d data set

- add a y-axis which is x^2

- add the y coordinate to each x coordinate by squaring the x-coordinate.

- this turns the data into a 2-dimensional dataset and we can now add a support vector classifier line

- used for when there isn't a clear delineation between 1 class and another in the current dimension

- requires data with a relatively low dimension

### Polynomial Kernel Function

- in order to make the mathematics possible, SVMs use something called kernel functions to systematically
find support vector classifiers in higher dimensions

- The polynomial kernel function has a parameter d, which stands for the degree of the polynomial

- when d=1 the polynomial kernel computes the relationships between each pair of observations in the dimension

