# Genetic-Algorithm

Implementation of Genetic-Algorithm for solution finding (optimization)



Easy to use GA implementation. With parallel computing and info-prints. Simple and flexible for your optimal solution finding.



<img src="./logo.jpeg"></img>



### Usage

1. Get the code

   - Download the project and add to python module search path in your code
   
       ``````python
       import sys
       sys.path.insert(0, '../path_to_GA_py_dir')
       # ./ => this folder
       # ../ => supfolder
       ``````

    - **Or** pip install it (easier)

        ``````python
        pip install Simple-Genetic-Algorithm
        ``````

3. Import the class and helper function

   ``````python
   from genetic_algorithm import GA, get_random
   ``````

4. Create 2 functions and parameters

   ``````python
   class Example_GA(GA):
   
       def calculate_fitness(self, kwargs, params):
           # return here the fitness (how good the solution is)
           # as bigger as better!
           # hint: if you have a loss, you should propably just return -1*loss
           # example for sklearn model:
            model = RandomForestRegressor(n_estimators=params["n_estimators"], ...)
     		 model = model.fit(kwargs["X_train"], kwargs["y_train"])
           # predict
           y_pred = model.predict(kwargs["X_test"])
           # calc mean absolute error loss
           y_true = np.array(kwargs["y_test"])
           y_pred = np.array(y_pred)
           return - np.mean(np.abs(y_true - y_pred))
   
       def get_random_value(self, param_key):
           if param_key == "name_of_parameter_1":
               return get_random(10, 1000)
           elif param_key == "name_of_parameter_2":
               return get_random(["something_1", "something_2", "something_3"])
           elif param_key == "name_of_parameter_3":
               return get_random(0.0, 1.0)
           
           ...
   
   parameters = ["n_estimators", "criterion", "max_depth", "max_features", "bootstrap"]
   ``````

5. Create and run genetic algorithm and pass the input, which will be used in the calculate_fitness function (in kwargs variable)

   ``````python
   optimizer = Example_GA(generations=10, population_size=15, mutation_rate=0.3, list_of_params=parameters)
   optimizer.optimize(X_train=X_train, y_train=y_train, X_test=X_dev, y_test=y_dev)
   ``````



Short explanation:<br>The **kwargs** are the inputs of optimize-method. These are the values which are needed to calculate the fitness. Maybe you can calculate the fitness without them, depending on what you are optimizing.<br>The **list of parameters** are the gene/the solution, so the parameters which are changed and optimized.<br>The **get_random_value** method return a random value for a given parameter, so that the solutions can be initialized and mutated.



### Examples
- <a href="./example.ipynb">Regression with RandomForrestRegressor</a>
- <a href="./example_2.ipynb">Knapsack problem</a>

<!--
<div style="border: 1px solid black; padding: 10px;">
    <iframe src="example.html" style="width:100%; height:400px;"></iframe>
</div> 
-->

<!--
<div style="border: 1px solid black; padding: 10px;">
    <iframe src="example_2.html" style="width:100%; height:400px;"></iframe>
</div>
-->

### License

Feel free to use it. Of course you don't have to name me in your code :)

-> It is a copy-left license

For all details see the license file.





