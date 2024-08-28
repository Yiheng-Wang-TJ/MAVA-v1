import time
from PIL import Image
# from IPython.display import Image
import google.generativeai as genai
from requests.exceptions import ReadTimeout
import requests
from google.api_core.exceptions import GoogleAPIError

class BaseConversation:
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

    def __init__(self, retrieved_data, query, prompt, storage_path):
        self.retrieved_data = retrieved_data
        self.query = query
        self.prompt = prompt
        self.storage_path = storage_path

    def generate_response(self, other_logger, result_logger, key):
        genai.configure(api_key=key, transport='rest')

        start = time.perf_counter()
        self.prompt = [self.prompt]
        self.prompt.append("Question:" + self.query)
        for s in self.retrieved_data:
            image_path = "{}/{:>06d}.png".format(self.storage_path, s)
            image = Image.open(image_path)
            width, height = image.size
            if width > 1960:
                new_width = 1960
                aspect_ratio = height / width
                new_height = int(new_width * aspect_ratio)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.prompt.append(image)
        end = time.perf_counter()
        other_logger.info('content:%s', self.prompt)
        other_logger.info('generate prompt cost:%s s', end - start)
        
        generate_costs = []
        results = []
        for r_i in range(1):
            start = time.perf_counter()
            pred = self.make_request(self.prompt, other_logger)
            result_logger.info('pred:%s', pred)
            end = time.perf_counter()

            other_logger.info('generate response cost:%s s', end - start)
            generate_costs.append(end - start)
            results.append(pred)
            print('generate cost: ', end - start)
            print(pred)
        return pred

    def make_request(self, content, other_logger):
        try:
            model = genai.GenerativeModel('gemini-1.5-pro-latest', safety_settings=self.safety_settings)
            response = model.generate_content(content)
            other_logger.info('response:%s', response)
            try:
                # Check if 'candidates' list is not empty
                if response.candidates:
                    # Access the first candidate's content if available
                    if response.candidates[0].content.parts:
                        generated_text = response.candidates[0].content.parts[0].text
                    else:
                        print("No generated text found in the candidate.")
                else:
                    print("No candidates found in the response.")
                    generated_text = 'No candidates found in the response.'
            except (AttributeError, IndexError) as e:
                print("Error:", e)
                generated_text = repr(e)
            return generated_text
        except ReadTimeout as e:
            print("Timeout error occurred:", e)
            other_logger.info('Timeout error occurred:%s', e)
            print("Retrying request...")
            return self.make_request(content, other_logger)
        except requests.exceptions.ConnectionError as e:
            print("Connection Error:", e)
            other_logger.info('Timeout error occurred:%s', e)
            print("Retrying request...")
            return self.make_request(content, other_logger)
        except genai.types.generation_types.BlockedPromptException as e:
            print(f"Prompt blocked due to: {e}")
            return self.make_request(content, other_logger)
        except requests.exceptions.HTTPError as err:
            if response.status_code == 400:
                print("Error 400: Bad request. The request was invalid.")
            else:
                print(f"HTTP error occurred: {err}")
            return self.make_request(content, other_logger)
        except GoogleAPIError as e:
            if e.code == 429:
                print("ResourceExhausted: ")
                time.sleep(10)
                return self.make_request(content, other_logger)
            else:
                print("An unexpected Google API error occurred: ", e)
                other_logger.info('An unexpected Google API error occurred:%s', e)
                return repr(e)
                # return make_request(content, other_logger)
        except Exception as e:
            print("An unexpected error occurred: ", e)
            return self.make_request(content, other_logger)
