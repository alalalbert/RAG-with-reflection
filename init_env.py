import os
import getpass

# run this if you need to update the environment variables

needed_env_vars = [
    "PINECONE_API_KEY", "ANTHROPIC_API_KEY", "SERPAPI_API_KEY",
    "COHERE_API_KEY"
]
'''
def check_and_request_env_var(var_name):
    if os.environ.get(var_name) is None:
        # Environment variable is not set. Ask the user for its value.
        print(f"{var_name} is not set. Please provide its value.")
        os.environ[var_name] = getpass.getpass(f"Enter {var_name}: ")
    else:
        # Environment variable is already set. 
        print(f"{var_name} is set.")
'''

for var_name in needed_env_vars:
  print(f"{var_name} needs to be set, please enter it now.")
  os.environ[var_name] = getpass.getpass(f"Enter {var_name}: ")
