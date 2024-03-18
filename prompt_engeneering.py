from transformers import AutoTokenizer, pipeline
import torch

hf_token = "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"
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model)
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1,
    token=hf_token
)

prompt =""" <s>[INST] <<SYS>>
As an AI specializing in software library recommendations for Android applications, provide recommendations exclusively in the Maven format: groupId:artifactId:version. Here is an example of how recommendations should be formatted:
- Example: com.fasterxml.jackson.core:jackson-databind:2.9.8

Your tasks are to:
- Provide recommendations exclusively in the Maven format as shown in the example.
- Ensure recommendations are suitable for an Android application focused on location services and reactive programming with RxJava.
- Maintain an unbiased approach based solely on technical merit and project compatibility.
- Use the context of the 'RxLocation' project for specific library recommendations.
- Utilize 'LATEST' for versions if specifics are unknown.
<</SYS>>

# Project Description
The project, 'RxLocation', is an Android library for reactive location updates using RxJava 2. The library is now deprecated, and users are advised to switch to the 'CoLocation' library.

# Existing Dependencies
The project's Maven dependencies include:
- io.reactivex.rxjava2:rxandroid
- com.google.android.gms:play-services-location
- com.android.support:appcompat-v7
- com.android.support:design
- junit:junit
- org.mockito:mockito-core

# Library Recommendation Request
With the context of the 'RxLocation' project, identify libraries to aid in the development of Android applications that utilize location services and reactive programming patterns. Recommendations must be in the Maven format and ensure compatibility with the existing dependencies.[/INST]</s>
"""

sequences = llama_pipeline(
    prompt,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    # max_length=250,  # Increased max_length
    truncation=True,  # Enable truncation
    # Alternatively, specify max_new_tokens to control the length of the generated portion specifically
    max_new_tokens=400  # This would generate up to 150 new tokens beyond the prompt
)

# Print the generated recommendations
for seq in sequences:
    print(f"Result: {seq['generated_text']}")