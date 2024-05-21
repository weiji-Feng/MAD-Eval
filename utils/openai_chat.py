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


text = """下面代碼的\quad沒有作用，爲什麼：
\begin{table*}[!htbp]
    \centering
    \small
    \begin{tabularx}{\textwidth}{XX}
        \toprule
        \multicolumn{2}{p{15.6cm}}{\textbf{Instruction:} \newline How can Dijkstra's algorithm be used to find the shortest path between two nodes in a graph?} \\ 
        \midrule

        \textbf{OpenChat-3.5 Response:\newline} 
        Dijkstra's algorithm is a popular algorithm used for finding the shortest path between two nodes in a weighted graph. Here's how it works:\newline 
        1. Initialize the algorithm:\newline 
        $\quad$- Create a set of unvisited nodes, and assign a tentative distance value to the starting node (usually 0) and to all other nodes (infinity).\newline 
        2. Select the unvisited node with the smallest tentative distance, and mark it as visited.\newline 
        3. Update the tentative distances of the neighboring nodes of the visited node. For each neighbor, calculate the new tentative distance by adding the distance from the starting node to the visited node to the distance from the visited node to the neighbor. If the new tentative distance is less than the current tentative distance of the neighbor, update the neighbor's tentative distance.\newline 
        4. Repeat steps 2 and 3 until all nodes have been visited or the target node has been visited.\newline 
        5. The shortest path from the starting node to the target node can be reconstructed by backtracking from the target node to the starting node, following the nodes with the smallest tentative distance at each step.\newline 
        Dijkstra's algorithm is guaranteed to find the shortest path in a graph with non-negative edge weights. It is particularly efficient for sparse graphs, but can be slow for dense graphs.\newline 
        To implement Dijkstra's algorithm, you can use the following Python code:\newline

        \parbox{\linewidth}{
            \texttt{import heapq\\
            def dijkstra(graph, start, target):\\
            \quad unvisited = set(graph.keys())\\
            \quad distances = \{node: float('inf') for node in unvisited\}\\
            \quad distances[start] = 0\\
            \quad path = \{\}\\
            \\
            \quad while unvisited:\\
            \quad \quad current\_node = min(unvisited, key=distances.get)\\
            \quad \quad if current\_node == target:\\
            \quad \quad \quad break\\
            \quad \quad unvisited.remove(current\_node)\\
            \\
            \quad \quad for neighbor, weight in graph[current\_node].items():\\
            \quad \quad \quad distance = distances[current\_node] + weight\\
            \quad \quad \quad if distance < distances[neighbor]:\\
            \quad \quad \quad \quad distances[neighbor] = distance\\
            \quad \quad \quad \quad path[neighbor] = current\_node\\
            \\
            \quad path\_to\_target = [target]\\
            \quad while path[path\_to\_target[-1]] is not None:\\
            \quad \quad path\_to\_target.append(path[path\_to\_target[-1]])\\
            \\
            \quad return distances[target], path\_to\_target[::-1]\\
            }
        }

        This function takes a graph (represented as a dictionary of nodes to their neighbors and weight), a starting node, and a target node. It returns the shortest path distance from the starting node to the target node, as well as the path itself (a list of nodes).
        & 
        \textbf{GPT-4-1106-preview Response:\newline}  
        I'd be happy to help you draft your inaugural speech, {\color{red}but I'll need some additional information}, such as:\newline 
        1. The position or role you are inaugurating.\newline 
        2. The audience you will be addressing (e.g., community members, professionals, students).\newline 
        3. Key themes or issues you wish to address.\newline 
        4. Your vision and goals in your new role.\newline 
        5. Any specific achievements or acknowledgments you want to include.\newline 
        6. The tone you want to set (e.g., hopeful, determined, conciliatory).\newline 
        With that information, I can help you create a draft that is tailored to your needs and the occasion. If you provide me with those details, I'll craft a speech for you. If you're not sure about some of the points, let me know, and I can offer you some general guidance that you can refine to fit your situation.
        \\ \midrule
        \multicolumn{2}{X}{\textbf{Human Comparison Result}: OpenChat-3.5}\\
        \bottomrule
    \end{tabularx}
    % \label{tab:addlabel}%
    \caption{Case study in \textit{writing} scenario. GPT-4-1106-preview exhibited a sense of 'laziness' in its responses, failing to address the demands of the instruction, despite the instruction lacking specific details.}
\end{table*}%

"""

if __name__ == "__main__":

    chat = OpenAIChat(model_name='gpt-4-0125-preview', max_tokens=4096, temperature=0.7, top_p=0.90)

    
    predictions = asyncio.run(chat.async_run(
        messages_list=[
            [
                {
                    "role": "user",
                    "content": text
                    # "content": prompt.format(question=d['instruction']+'\n'+d['input'], answer=d['output']),
                    # "content": "show either 'ab' or '['a']'. Do not do anything else."
                }
            ]
        ],
        expected_type=List,
    ))

    for pred in predictions:
        print(pred)
    
    # f = open('/home/dky/khfeng/datasets/instruction/multichoice_en.json', 'w')
    # json.dump(data, f, indent=4)
