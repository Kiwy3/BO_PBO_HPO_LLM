from algorithm import BO_HPO

if __name__ == "__main__":

    g=20
    bo = BO_HPO(config="./config.json")
    bo.init(g)
    print(bo.X,"\n", bo.Y, bo.bounds)
    bo.run(n=50)