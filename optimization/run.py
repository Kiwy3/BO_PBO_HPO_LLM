from algorithm import BO_HPO

if __name__ == "__main__":

    g=10
    bo = BO_HPO()
    bo.init()
    print(bo.X,"\n", bo.Y, bo.bounds)
    bo.run(n=20)