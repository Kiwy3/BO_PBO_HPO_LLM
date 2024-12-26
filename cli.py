import sys
from optimization_v2.experiments import exp10


if len(sys.argv) < 2:
    print("No argument provided")
    sys.exit(1)

arg = sys.argv[1]
print(f"Received argument: {arg}")


if arg == "exp10" : 
    exp10.main()
else : 
    print(f"Unknown argument : {arg}")