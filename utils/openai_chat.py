# the async version is adapted from https://gist.github.com/neubig/80de662fb3e225c18172ec218be4917a

import os
import openai
import ast
import asyncio
from typing import List


class OpenAIChat():
    def __init__(self, model_name='gpt-3.5-turbo', max_tokens=2500, temperature=0.5, top_p=1, request_timeout=60, stop=None, request_id=None, json_mode=False):
        self.config = {
            'model_name': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'request_timeout': request_timeout,
            'request_id': request_id,
            'stop': stop,
            'json_mode': json_mode
            }
        if "gpt" in model_name or 'embedding' in model_name:
            openai.api_key = os.environ['OPENAI_API_KEY']
            openai.api_base = os.environ['OPENAI_API_BASE']
        else:
            openai.api_key = "EMPTY"
            openai.api_base = "http://127.0.0.1:8333/v1"
        
    
    def _boolean_fix(self, output):
        return output.replace("true", "True").replace("false", "False")

    def _type_check(self, output, expected_type):
        try:
            output_eval = ast.literal_eval(output)
            if not isinstance(output_eval, expected_type):
                return None
            return output_eval
        except:
            return None

    async def dispatch_openai_requests(
        self,
        messages_list,
    ) -> List[str]:
        """Dispatches requests to OpenAI API asynchronously.
        
        Args:
            messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        Returns:
            List of responses from OpenAI API.
        """
        async def _request_with_retry(messages, retry=3):
            for try_i in range(retry):
                try:
                    # for text embedding models
                    if "embedding" in self.config['model_name']:
                        response = await openai.Embedding.acreate(
                            model=self.config['model_name'],
                            input=messages,
                        )
                    else:
                        # for chat models
                        if self.config['json_mode'] == True:
                            response = await openai.ChatCompletion.acreate(
                                model=self.config['model_name'],
                                response_format={'type': 'json_object'},
                                messages=messages,
                                max_tokens=self.config['max_tokens'],
                                temperature=self.config['temperature'],
                                top_p=self.config['top_p'],
                                request_timeout=self.config['request_timeout'],
                                stop=self.config['stop'],
                            )
                        else:
                            response = await openai.ChatCompletion.acreate(
                                model=self.config['model_name'],
                                messages=messages,
                                max_tokens=self.config['max_tokens'],
                                temperature=self.config['temperature'],
                                top_p=self.config['top_p'],
                                request_timeout=self.config['request_timeout'],
                                stop=self.config['stop'],
                            )
                    return response
                except openai.error.InvalidRequestError as e:
                    print(e)
                    print(f'Retry {try_i+1} Invalid request error, waiting for 3 second...')
                    await asyncio.sleep(3)
                    # return None
                except openai.error.RateLimitError:
                    print(f'Retry {try_i+1} Rate limit error, waiting for 40 second...')
                    await asyncio.sleep(40)
                except openai.error.APIError:
                    print(f'Retry {try_i+1} API error, waiting for 5 second...')
                    await asyncio.sleep(5)
                except openai.error.Timeout:
                    print(f'Retry {try_i+1} Timeout error, waiting for 10 second...')
                    await asyncio.sleep(10)
                except openai.error.APIConnectionError:
                    print(f'Retry {try_i+1} API connection error, waiting for 10 second...')
                    await asyncio.sleep(10)
                except openai.error.ServiceUnavailableError:
                    print(f'Retry {try_i+1} Service unavailable error, waiting for 3 second...')
                    await asyncio.sleep(3)
            return None

        async_responses = [
            _request_with_retry(messages)
            for messages in messages_list
        ]

        return await asyncio.gather(*async_responses)
    
    async def async_run(self, messages_list, expected_type):
        retry = 10
        responses = [None for _ in range(len(messages_list))]
        messages_list_cur_index = [i for i in range(len(messages_list))]

        while retry > 0 and len(messages_list_cur_index) > 0:
            # print(f'{retry} retry left...')
            messages_list_cur = [messages_list[i] for i in messages_list_cur_index]
            
            predictions = await self.dispatch_openai_requests(
                messages_list=messages_list_cur,
            )

            if "embedding" in self.config['model_name']:
                preds = [prediction['data'][0]['embedding'] if prediction is not None else None for prediction in predictions]
            else:
                preds = [prediction['choices'][0]['message']['content'] if prediction is not None else None for prediction in predictions]

            finised_index = []
            for i, pred in enumerate(preds):
                if pred is not None:
                    responses[messages_list_cur_index[i]] = pred
                    finised_index.append(messages_list_cur_index[i])
            
            messages_list_cur_index = [i for i in messages_list_cur_index if i not in finised_index]
            
            retry -= 1
        
        return responses



if __name__ == "__main__":

    chat = OpenAIChat(model_name='gpt-4-0125-preview', max_tokens=4096, temperature=0.7, top_p=0.90)

    
    predictions = asyncio.run(chat.async_run(
        messages_list=[
            [
                {
                    "role": "user",
                    "content": "show either 'ab' or '['a']'. Do not do anything else."
                }
            ]
        ],
        expected_type=List,
    ))

    for pred in predictions:
        print(pred)
    
    # f = open('/home/dky/khfeng/datasets/instruction/multichoice_en.json', 'w')
    # json.dump(data, f, indent=4)
