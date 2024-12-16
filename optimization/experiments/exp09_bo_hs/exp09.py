from algorithm import BO_HPO

if __name__ == "__main__":

    g=15
    import os; print(os.getcwd())
    bo = BO_HPO(config_file="optimization/experiments/exp09_bo_hs/config.json")
    
    bo.init(g)
    print(bo.X,"\n", bo.Y, bo.bounds)
    bo.run(
        n = bo.experiment["calls"]
    )