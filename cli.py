import sys
from optimization_v2.experiments import exp10, exp11, exp12, exp13


if len(sys.argv) < 2:
    print("No argument provided")
    sys.exit(1)

arg = sys.argv[1]
print(f"Received argument: {arg}")


if arg == "exp10" : 
    exp10.main()
elif arg == "exp11" : 
    exp11.main()
elif arg == "exp12" : 
    exp12.main()
elif arg == "exp13" : 
    exp13.main()
elif arg == "exp13_bis" : 
    exp13.bis()
else : 
    print(f"Unknown argument : {arg}")