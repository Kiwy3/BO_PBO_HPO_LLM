from algorithm import BO_HPO

if __name__ == "__main__":

    g=10
    bo = BO_HPO(LHS_g=g)
    print(bo.X,"\n", bo.Y, bo.bounds)
    bo.run(n=50)