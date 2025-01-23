import sys
from hpo.experiments import bo, soo, bamsoo, sampling


if len(sys.argv) < 2:
    print("No argument provided")
    sys.exit(1)

arg = sys.argv[1]
print(f"Received argument: {arg}")

if arg == "sampling" : 
    sampling.main()
elif arg == "bo" : 
    bo.main()
elif arg == "soo" : 
    soo.main()
elif arg == "bamsoo" : 
    bamsoo.main()
else : 
    print(f"Unknown argument : {arg}")