from langchain_core.prompts import PromptTemplate, MessagesPlaceholder,AIMessagePromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
from operator import itemgetter
import json
import os
import sys
import yaml
# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.LocalEsStore import LocalElasticVectorStore
from utils.LocalEmbeddings import LocalEmbeddings
from chat.LocalQwenChat_tran import LocalQwenChat
from chain.BaseClass import PointsMarkResult

from langchain_openai import ChatOpenAI

from chain.ThinkJsonOutputParser import ThinkJsonOutputParser

# 获取大模型配置
with open("config/config2.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
llm_config = config['llm']

# 规则后处理
def process_after_retrieve_rule(docs):
    rules = []
    for doc in docs:
        rules.append({
            "rule":doc.page_content
        })
    return rules

# 样例后处理
def process_after_retrieve_example(docs):
    examples = []
    for doc in docs:
        examples.append({
            "text":doc.page_content,
            "project_type":doc.metadata.get('project_type',''),
            "secret_point": doc.metadata.get('secret_point', ''),
            "result":doc.metadata.get('points','[]')
        })
    return examples

# 密点标注后处理
def process_after_point_secret_point(secret_point_li):
    
    secret_point_str = ""
    secret_point_li = secret_point_li[0]['secret_type']
    for i in range(0, len(secret_point_li)):
        if i != len(secret_point_li)-1:
            secret_point_str += str(secret_point_li[i]) + ";"
        else:
            secret_point_str += str(secret_point_li[i])
    return secret_point_str

class CombineChain:
    def __init__(self, config_example: dict, config_rule: dict):
        self.config_example = config_example
        self.config_rule = config_rule

        # 样例检索器
        self.example_retriever = LocalElasticVectorStore(
            index_name = config_example.get('index_name'),
            embedding=LocalEmbeddings(),
            config = config_example
        ).as_retriever(config_example.get('retriever_name'))
        # 规则检索器
        self.rule_retriever = LocalElasticVectorStore(
            index_name = config_rule.get('index_name'),
            embedding=LocalEmbeddings(),
            config = config_rule
        ).as_retriever(config_rule.get('retriever_name'))
        self.__LCEL()
    
    def _prompt(self, rules, examples, template_format="jinja2"):
        # 任务描述模板
        task_template = """你是一名保密工作专家，请你根据提供的基础信息和具体的标注规则、标注样例，标注文本中的密点和密级。
            1、密点是指具有保密价值的信息内容，其可以划分为四个等级：一级、二级、三级、未涉密，一级保密价值最高；
            2、段落中可能含有零个、一个或多个密点，将所有结果放入“[]”中返回，如果没有密点，则返回“[]”；
            3、输入的信息为项目类型（使用<项目类型></项目类型>给出）和需要识别的文本（使用<文本></文本>给出），项目类型为需要识别文本所属的项目，是确定密级的重要依据。
            4、涉密点包括：项目类型（描述研究项目的类型）、项目名称（一般是具体的研发项目）、产品及型号（特别注意，这里的产品及型号仅指带有型号的船舶或舰艇，其他带有型号的系统、机器等都不算，如“C789型船舶”）、发明人、审查员、任务背景（一般是指用于某些军用或者民用领域等）、用途（一般指设备或技术用于某某型号的船舶或舰艇）、主要设备组成（一般是指包括哪些部分，具体到哪些配件）、主要性能指标（一般是带有关键参数的信息）、关键技术（一般是指具有完整流程的技术步骤或者原理方法）。
            5、在标注过程中，你需要先判断文本中是否包含涉密点，再根据项目类型等信息对其进行密级的判断。
            6、直接输出密点标注结果，不要包含解析过程。
        
        以下是一些具体的标注样例（使用<样例></样例>给出）：\n\n
        <样例>
        
        """

        rule_start = "</样例>\n\n以下是一些具体的标注规则（使用<规则></规则>给出）：\n\n<规则>\n"
        rule_end = "</规则>\n\n"

        # 样例
        example_prompt =PromptTemplate.from_template('<文本>{text}</文本>\n<项目类型>{project_type}</项目类型>\n<涉密点>{secret_point}</涉密点>\n<密点标注结果>{result}</密点标注结果>')

        example_template = ''
        for example in examples:
            example_template += example_prompt.format(**example) + '\n\n'
        
        # 规则
        rule_prompt =PromptTemplate.from_template('规则{{index}}:{{text}}',template_format='jinja2')

        rule_template = ''
        for index,rule in enumerate(rules):
            rule_template += rule_prompt.format(index=index+1,text=rule.get('rule','')) + '\n\n'

        # 文本模板
        suffix_template = "回答格式：{{format}}\n请你遵照规则、样例和回答格式对以下文本进行标注。\n\n<文本>{{text}}</文本>\n<项目类型>{{project_type}}</项目类型>\n<涉密点>{{secret_point}}</涉密点>"

        # 总模板
        _prompt =  PromptTemplate.from_template(task_template + example_template + rule_start + rule_template + rule_end + suffix_template, template_format="jinja2")
        
        return _prompt

    def prompt(self, _dict):
        prompt_template = self._prompt(_dict.get("rules"), _dict.get("examples"))

        _prompt = prompt_template.format(
                        text=_dict.get("text", ""),
                        project_type=_dict.get("project_type",''),
                        secret_point=_dict.get("secret_point", ""),
                        format = JsonOutputParser(pydantic_object=PointsMarkResult).get_format_instructions()
        )

        return _prompt

    def prompt_secret_type(self, text):
        # 任务描述模板
        task_template = """涉密点抽取任务说明：
                            1、涉密点包括：项目类型（描述研究项目的类型）、项目名称（一般是具体的研发项目）、产品及型号（特别注意，这里的产品及型号仅指带有型号的船舶或舰艇，其他带有型号的系统、机器等都不算，如“C789型船舶”）、发明人（不包括代理人和专利权人）、审查员、任务背景（一般是指用于某些军用或者民用领域等）、用途（一般指设备或技术用于某某型号的船舶或舰艇）、主要设备组成（一般是指包括哪些部分，具体到哪些配件）、主要性能指标（一般是带有关键参数的信息）、关键技术（一般是指具有完整流程的技术步骤或者原理方法）。
                            2、判断文本中是否包括上述的涉密点，一个段落可能包括零个、一个或多个涉密点。如果有描述则输出对应的涉密点，注意，只需要输出涉密点名称即可，如“主要设备组成”、“产品及型号”等等。不能根据内容推断涉密点，也不能输出上述涉密点以外的其他内容。
                            3、注意事项：
                            （1）产品及型号仅指带有型号的船舶或舰艇，其他带有型号的系统、机器等不算。例如，“C789型船舶”符合条件，而“TXR888型大功率变压器”不符合条件。
                            （2）主要设备组成应具体到哪些配件或部分。
                            （3）关键技术描述单个或完整流程的技术步骤以及原理方法。
                            4、输出格式为json格式，即```{ "secret_type": [涉密点]}```。如果段落中没有包括涉密点，则输出```{ "secret_type": []}```
                            
                        """
        # 文本模板
        suffix_template = '''请你遵照上述规则对以下文本进行标注。\n\n<文本>{{text}}</文本>'''
        # 拼接
        template =  PromptTemplate.from_template(task_template + suffix_template, template_format="jinja2")
        _prompt = template.format(text=text)
        
        return _prompt

    def __get_llm(self):
        return ChatOpenAI(
            model_name=llm_config['model_name'],
            openai_api_key=llm_config['api_key'],
            openai_api_base=llm_config['api_base'],
            streaming=llm_config['streaming'],
            extra_body={"enable_thinking": llm_config['enable_thinking']},
            temperature=llm_config['temperature'],
            top_p=llm_config['top_p']
        )

    def __LCEL(self):
        self.chain = (RunnableParallel(
                                rules=itemgetter('project_type') | self.rule_retriever | process_after_retrieve_rule,
                                examples=RunnableLambda(lambda input: str(input)) | self.example_retriever | process_after_retrieve_example,
                                secret_point = RunnableLambda(lambda x: x['text'])  | self.prompt_secret_type | self.__get_llm() | ThinkJsonOutputParser() | process_after_point_secret_point, 
                                text = itemgetter('text'),
                                project_type = itemgetter('project_type')
                                ) 
                        | self.prompt 
                        | self.__get_llm()
                        | ThinkJsonOutputParser())

    def call(self, text, project_type):
        return self.chain.invoke({'text':text,'project_type':project_type})


class CombineChain2:
    def __init__(self, config_example: dict, config_rule: dict, model_name: str):
        self.config_example = config_example
        self.config_rule = config_rule
        self.model_name = model_name

        # 样例检索器
        self.example_retriever = LocalElasticVectorStore(
            index_name = config_example.get('index_name'),
            embedding=LocalEmbeddings(),
            config = config_example
        ).as_retriever(config_example.get('retriever_name'))
        # 规则检索器
        self.rule_retriever = LocalElasticVectorStore(
            index_name = config_rule.get('index_name'),
            embedding=LocalEmbeddings(),
            config = config_rule
        ).as_retriever(config_rule.get('retriever_name'))
        self.__LCEL()
    
    def _prompt(self, rules, examples, template_format="jinja2"):
        # 任务描述模板
        task_template = """你是一名保密工作专家，请你根据提供的基础信息和具体的标注规则、标注样例，标注文本中的密点和密级。
            1、密点是指具有保密价值的信息内容，其可以划分为四个等级：一级、二级、三级、未涉密，一级保密价值最高；
            2、段落中可能含有零个、一个或多个密点，将所有结果放入“[]”中返回，如果没有密点，则返回“[]”；
            3、输入的信息为项目类型（使用<项目类型></项目类型>给出）和需要识别的文本（使用<文本></文本>给出），项目类型为需要识别文本所属的项目，是确定密级的重要依据。
            4、涉密点包括：项目类型（描述研究项目的类型）、项目名称（一般是具体的研发项目）、产品及型号（特别注意，这里的产品及型号仅指带有型号的船舶或舰艇，其他带有型号的系统、机器等都不算，如“C789型船舶”）、发明人、审查员、任务背景（一般是指用于某些军用或者民用领域等）、用途（一般指设备或技术用于某某型号的船舶或舰艇）、主要设备组成（一般是指包括哪些部分，具体到哪些配件）、主要性能指标（一般是带有关键参数的信息）、关键技术（一般是指具有完整流程的技术步骤或者原理方法）。
            5、在标注过程中，你需要先判断文本中是否包含涉密点，再根据项目类型等信息对其进行密级的判断。
            6、直接输出密点标注结果，不要包含解析过程。
        
        以下是一些具体的标注样例（使用<样例></样例>给出）：\n\n
        <样例>
        
        """

        rule_start = "</样例>\n\n以下是一些具体的标注规则（使用<规则></规则>给出）：\n\n<规则>\n"
        rule_end = "</规则>\n\n"

        # 样例
        example_prompt =PromptTemplate.from_template('<文本>{text}</文本>\n<项目类型>{project_type}</项目类型>\n<涉密点>{secret_point}</涉密点>\n<密点标注结果>{result}</密点标注结果>')

        example_template = ''
        for example in examples:
            example_template += example_prompt.format(**example) + '\n\n'
        
        # 规则
        rule_prompt =PromptTemplate.from_template('规则{{index}}:{{text}}',template_format='jinja2')

        rule_template = ''
        for index,rule in enumerate(rules):
            rule_template += rule_prompt.format(index=index+1,text=rule.get('rule','')) + '\n\n'

        # 文本模板
        suffix_template = "回答格式：{{format}}\n请你遵照规则、样例和回答格式对以下文本进行标注。\n\n<文本>{{text}}</文本>\n<项目类型>{{project_type}}</项目类型>\n<涉密点>{{secret_point}}</涉密点>"

        # 总模板
        _prompt =  PromptTemplate.from_template(task_template + example_template + rule_start + rule_template + rule_end + suffix_template, template_format="jinja2")
        
        return _prompt

    def prompt(self, _dict):
        prompt_template = self._prompt(_dict.get("rules"), _dict.get("examples"))

        _prompt = prompt_template.format(
                        text=_dict.get("text", ""),
                        project_type=_dict.get("project_type",''),
                        secret_point=_dict.get("secret_point", ""),
                        format = JsonOutputParser(pydantic_object=PointsMarkResult).get_format_instructions()
        )
        # print(_prompt)
        return _prompt

    def prompt_secret_type(self, text):
        # 任务描述模板
        task_template = """涉密点抽取任务说明：
                            1、涉密点包括：项目类型（描述研究项目的类型）、项目名称（一般是具体的研发项目）、产品及型号（特别注意，这里的产品及型号仅指带有型号的船舶或舰艇，其他带有型号的系统、机器等都不算，如“C789型船舶”）、发明人（不包括代理人和专利权人）、审查员、任务背景（一般是指用于某些军用或者民用领域等）、用途（一般指设备或技术用于某某型号的船舶或舰艇）、主要设备组成（一般是指包括哪些部分，具体到哪些配件）、主要性能指标（一般是带有关键参数的信息）、关键技术（一般是指具有完整流程的技术步骤或者原理方法）。
                            2、判断文本中是否包括上述的涉密点，一个段落可能包括零个、一个或多个涉密点。如果有描述则输出对应的涉密点，注意，只需要输出涉密点名称即可，如“主要设备组成”、“产品及型号”等等。不能根据内容推断涉密点，也不能输出上述涉密点以外的其他内容。
                            3、注意事项：
                            （1）产品及型号仅指带有型号的船舶或舰艇，其他带有型号的系统、机器等不算。例如，“C789型船舶”符合条件，而“TXR888型大功率变压器”不符合条件。
                            （2）主要设备组成应具体到哪些配件或部分。
                            （3）关键技术描述单个或完整流程的技术步骤以及原理方法。
                            4、输出格式为json格式，即```{ "secret_type": [涉密点]}```。如果段落中没有包括涉密点，则输出```{ "secret_type": []}```
                            
                        """
        # 文本模板
        suffix_template = '''请你遵照上述规则对以下文本进行标注。\n\n<文本>{{text}}</文本>'''
        # 拼接
        template =  PromptTemplate.from_template(task_template + suffix_template, template_format="jinja2")
        _prompt = template.format(text=text)
        
        return _prompt

    def __get_llm(self):
        """根据模型名称选择 ChatOpenAI"""
        return ChatOpenAI(
            model_name=self.model_name,
            openai_api_key="EMPTY",
            openai_api_base='http://115.156.114.150:8011/v1',
            # 通过 extra_body 设置 enable_thinking = False 关闭思考过程
            extra_body={"enable_thinking": False},
            streaming=True,
            temperature=0.01,
            top_p=0.01
        )

    def __LCEL(self):
        self.chain = (RunnableParallel(
                                rules=itemgetter('project_type') | self.rule_retriever | process_after_retrieve_rule,
                                examples=RunnableLambda(lambda input: str(input)) | self.example_retriever | process_after_retrieve_example,
                                secret_point = RunnableLambda(lambda x: x['text'])  | self.prompt_secret_type | self.__get_llm() | ThinkJsonOutputParser() | process_after_point_secret_point, 
                                text = itemgetter('text'),
                                project_type = itemgetter('project_type')
                                ) 
                        | self.prompt 
                        | self.__get_llm()
                        | ThinkJsonOutputParser())

    def retirver_test(self, text, project_type):
        chain_test = self.prompt_secret_type | self.__get_llm() | JsonOutputParser() | RunnableLambda(lambda x:x['secret_type'])
        # print(chain_test.invoke({'text':text}))
        chain = (RunnableParallel(
                                rules=itemgetter('project_type') | self.rule_retriever | process_after_retrieve_rule,
                                examples=RunnableLambda(lambda input: str(input)) | self.example_retriever | process_after_retrieve_example,
                                secret_point = RunnableLambda(lambda x: x['text'])  | self.prompt_secret_type | self.__get_llm() | JsonOutputParser() | RunnableLambda(lambda x:x['secret_type']), 
                                text = itemgetter('text'),
                                project_type = itemgetter('project_type')
                                ) 
                        | self.prompt)
        
        prompt = chain.invoke({'text':text,'project_type':project_type})
        # print(prompt)

    def call(self, text, project_type):
        return self.chain.invoke({'text':text,'project_type':project_type})

if __name__ == "__main__":
    project_type = "国防基础科研项目"
    text = '''本发明是ZL623型超导线焊接装置研发项目的一部分，涉及一种超导线的焊接装置及焊接方法。
              焊接装置包括加热单元、焊接台板、隔热垫块和具有支撑体的压紧单元；所述焊接台板紧固在加热单元上；
              加热单元设有加热器和温度调节器；压紧单元安装在焊接台板上；压紧单元还包括有：
              安装在支撑体上用于对焊接头施压的压紧件，以及位于焊接台板与支撑体之间的隔热垫块。
              本发明的优点是：提供一种控温温区范围大、精度高，能够满足超导线焊接控温要求的超导线焊接装置及其方法，
              通过减小焊层厚度，增大焊接面积，减小焊接电阻，降低人为操作因素，提高焊接质量，降低焊接分散性。'''
    config_example = {
        'index_name':'train_paragraph_index_20241203',
        'retriever_name':'ParaRetriever',
        'bm25_top_k':10,
        'embs_top_k':3,
        'threshold':1.0,
        'query_field':'content',
    }
    config_rule = {
        'index_name':'rule_20241203',
        'retriever_name':'RuleRetriever',
        'bm25_top_k':5,
        'embs_top_k':3,
        'threshold':1.0,
        'query_field':'content',
    }
    chain = CombineChain(config_example, config_rule)
    # res1 = chain.retirver_test(text, project_type)
    # print("----------------")
    # print(res1)
    res = chain.call(text, project_type)
    print("----------------")
    print(res)
    # res = res.replace("'", '"').replace('```','').replace('json','')
    # res = json.loads(res)
    # print(res)