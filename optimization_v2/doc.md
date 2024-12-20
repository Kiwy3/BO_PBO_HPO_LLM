## SearchSpace

### Input
    mode : str ["base"] => use classic definition of search space
    savefile : str => file to save solution and results 

### Attributes
    savefile : path to save solution and results
    variables : dict of var
    center
    coef : coefficient of all dimensions, used in SOO

### Function
    base_init(self) ->None : configure variables with search space of article
    get_center(self) -> Solution : renvoie le center d'un point en tant que solution
    init_coef(self) -> None : compute coef for each variable
    add_variables(self, variables : dict) -> None : add a var from a dict
    section(K) -> [SearchSpace]*K : divide the Search space in K section, based on biggest size (use coef)
    get_solution(x) -> Solution : take list of value and return a solution


## Solution (Search Space)
Inherite from search space, especially to keep variables

### Input
    variables : dict of var => already fonctionnal variables
    savefile : str => inherite file from Search
    x : tuple of value

### Attributes



### Function

## var

### Input


### Attributes


### Function

# Eval

## var

### Input


### Attributes


### Function