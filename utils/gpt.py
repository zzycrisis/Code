import os
import time
from openai import OpenAI, AzureOpenAI


class ChatGPT:
    def __init__(self, model_name="gpt-3.5-turbo", configs=None, **kwargs) -> None:
        '''
        model_name: api version of a openai model (https://platform.openai.com/docs/models/overview)
        Additional kwargs for Azure OpenAI: api_type, api_key, base_url, api_version, deployment_name
        Priority: explicit kwargs > environment variables > defaults
        '''
        self.model_name = model_name

        # Resolve API configuration: explicit kwargs > environment variables > defaults
        api_type = kwargs.get("api_type") or os.getenv("API_TYPE", "openai")
        api_key = kwargs.get("api_key") or os.getenv("API_KEY")
        base_url = kwargs.get("base_url") or os.getenv("BASE_URL")
        api_version = kwargs.get("api_version") or os.getenv("API_VERSION")
        deployment_name = kwargs.get("deployment_name") or os.getenv("DEPLOYMENT_NAME")

        if api_type == "azure":
            if not api_version:
                raise ValueError(
                    "api_version is required for Azure OpenAI. "
                    "Pass it as api_version= or set the API_VERSION env var."
                )
            self.client = AzureOpenAI(
                azure_endpoint=base_url,
                api_key=api_key,
                api_version=api_version,
            )
            # On Azure, model= in chat.completions.create maps to deployment name.
            # Use deployment_name if given, otherwise fall through to model_name.
            if deployment_name:
                self.model_name = deployment_name
        else:
            # Standard OpenAI (backward-compatible path)
            self.client = OpenAI(
                base_url=base_url + "/v1",
                api_key=api_key,
            )

        self.config = {
                'n': 1,
                'temperature': 1,
                'top_p': 1
            }
        if configs is not None:
            self.config = configs
    
    def fit_message(self, msg, system_prompt=None):
        if system_prompt is not None:
            conversation = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": msg}
                    ]
        else:
            conversation = [
                        {"role": "user", "content": msg}
                    ]
        return conversation

    def query(self, msg, system_prompt=None, **kwargs):
        start = time.time()

        while True:
            try:
                raw_response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.fit_message(msg, system_prompt),
                            **kwargs)
                self.raw_response = raw_response

                return str(raw_response.choices[0].message.content)
            except Exception as e:
                print(e)

            current = time.time()
            if current - start > 60: return None
            time.sleep(10)
    
    def __call__(self, msg, system_prompt=None):
        return self.query(msg, system_prompt, **self.config)


# Example usage
# chatgpt = ChatGPT()
# print(chatgpt("Who are you?"))
