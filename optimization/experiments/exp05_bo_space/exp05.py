from algorithm import BO_HPO

if __name__ == "__main__":

    g=20
    import os; print(os.getcwd())
    bo = BO_HPO(config_file="optimization/experiments/exp05_bo_space/config.json")
    
    bo.init(g)
    print(bo.X,"\n", bo.Y, bo.bounds)
    bo.run(n=50)