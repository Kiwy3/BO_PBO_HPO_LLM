import sys
from optimization_v2.experiments import exp10, exp11, exp12, exp13
from optimization_v2.experiments import bo, soo, bamsoo, sampling


if len(sys.argv) < 2:
    print("No argument provided")
    sys.exit(1)

arg = sys.argv[1]
print(f"Received argument: {arg}")


if arg == "exp10" : 
    exp10.main()
if arg == "exp10_bis" : 
    exp10.bis()
elif arg == "exp11" : 
    exp11.main()
elif arg == "exp11_bis" : 
    exp11.bis()
elif arg == "exp12" : 
    exp12.main()
elif arg == "exp12_bis" : 
    exp12.bis()
elif arg == "exp13" : 
    exp13.main()
elif arg == "exp13_bis" : 
    exp13.bis()
elif arg == "exp13_ter" : 
    exp13.ter()

elif arg == "sampling" : 
    sampling.main()
elif arg == "lhs_bis" : 
    sampling.bis()
elif arg == "bo" : 
    bo.main()
elif arg == "bo_bis" : 
    bo.bis()
elif arg == "soo" : 
    soo.main()
elif arg == "bamsoo" : 
    bamsoo.main()
else : 
    print(f"Unknown argument : {arg}")