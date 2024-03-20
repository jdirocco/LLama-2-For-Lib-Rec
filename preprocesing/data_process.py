import pandas as pd
import ast
import re
data = pd.read_csv('./data/project_dependencies_readmes.csv')

# Since the README and dependencies are likely in string representation, I will convert them to a more usable format
# Decode the README contents from bytes literal to a string (assuming it's encoded in UTF-8)
data['readme'] = data['readme'].apply(lambda x: eval(x).decode('utf-8'))


# Additionally, we will remove any leading '#' from the dependencies as they seem to be non-standard
def clean_readme(readme):
    # Remove URLs
    readme = re.sub(r'http\S+|www.\S+', '', readme)
    
    # Remove emojis
    readme = readme.encode('ascii', 'ignore').decode('ascii')
    
    # Remove markdown URLs and images
    readme = re.sub(r'\[.*?\]\(.*?\)', '', readme)
    
    # Remove code blocks enclosed in triple backticks
    readme = re.sub(r'```.*?```', '', readme, flags=re.DOTALL)
    
    # Remove inline code snippets enclosed in single backticks
    readme = re.sub(r'`.*?`', '', readme)
    
    # Remove lines that start with bash script indicators `$`
    readme = re.sub(r'^\$\s.*', '', readme, flags=re.MULTILINE)
    
    # Remove HTML tags
    readme = re.sub(r'<.*?>', '', readme, flags=re.DOTALL)
    
    # Remove additional Markdown syntax elements (headers, lists, etc.)
    readme = re.sub(r'^[\#\*\-\>]+', '', readme, flags=re.MULTILINE)
    
    # Remove additional markdown formatting characters
    readme = re.sub(r'[\*_]', '', readme)
    
    # Remove excess whitespace between lines
    readme = re.sub(r'\n\s*\n', '\n\n', readme)
    
    # Remove any remaining syntax that might be considered code
    readme = re.sub(r'\s*\(.*?\)\s*\{.*?\}', '', readme, flags=re.DOTALL)
    
    # Remove any remaining XML configurations
    readme = re.sub(r'<\?xml.*?\?>', '', readme, flags=re.DOTALL)
    
    # Remove any remaining single-line code snippets
    readme = re.sub(r'^\s*\$.*$', '', readme, flags=re.MULTILINE)
    
    # Remove any remaining inline code snippets enclosed in single backticks
    readme = re.sub(r'`.*?`', '', readme)
    
    # Remove any remaining excess whitespace between lines
    readme = re.sub(r'\n\s*\n', '\n\n', readme)
    
    return readme.strip()


# Additionally, we will remove any leading '#' from the dependencies as they seem to be non-standard

def parse_dependencies(dependencies):
    try:
        # Convert string representation of list to an actual list
        dependencies_list = ast.literal_eval(dependencies)
        dependencies_list = [dep.lstrip('#').strip() for dep in dependencies_list]
        return dependencies_list
    except:
        return []
    

data['dependencies'] = data['dependencies'].apply(parse_dependencies)
data['readme'] = data['readme'].apply(clean_readme)
# Now, we can use the pipeline to generate the recommendations for each README
parsed_data_file_path = './data/processed/parsed_data.csv'
data.to_csv(parsed_data_file_path, index=False)
print(f"Data saved to {parsed_data_file_path}")