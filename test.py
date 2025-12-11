import pandas as pd
from tqdm import tqdm
import json
import sys
import os
# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics_calculate import *
from chain.ParaLabelChain2 import ParaLabelChain
from chain.CombineChain2 import CombineChain
from chain.RULEChain2 import RULEChain
from chain.RAGChain2 import RAGChain

from utils.data_process import merge_json, mark_instruction_error

# ---------------- 配置 ----------------
rule_config = {
        'index_name':'rule_20241203',
        'retriever_name':'RuleRetriever',
        'bm25_top_k':5,
        'embs_top_k':3,
        'threshold':1.0,
        'query_field':'content',
}
rag_config = {
    'index_name':'train_paragraph_index_20241203',
    'retriever_name':'ParaRetriever',
    'bm25_top_k':10,
    'embs_top_k':3,
    'threshold':1.0,
    'query_field':'content',
}
combine_example_config = {
    'index_name':'train_paragraph_index_20241203',
    'retriever_name':'ParaRetriever',
    'bm25_top_k':10,
    'embs_top_k':3,
    'threshold':1.0,
    'query_field':'content',
}
combine_rule_config = {
    'index_name':'rule_20241203',
    'retriever_name':'RuleRetriever',
    'bm25_top_k':5,
    'embs_top_k':3,
    'threshold':1.0,
    'query_field':'content',
}

# ---------------- 通用处理函数 ----------------
def process_para(res):
    try:
        if not isinstance(res,list):
            return [],'未涉密',1
        else:
            points_num_llm = len(res)
            if points_num_llm == 1:
                return res,res[0]['point_security_level'],0
            elif points_num_llm > 1:
                # 定义映射
                rank_mapping = {
                    '一级': 3,
                    '二级': 2,
                    '三级': 1
                }

                # 找出字段值最大的一行
                max_record = max(res, key=lambda x: rank_mapping.get(x['point_security_level'], 0))
                return res,max_record['point_security_level'],0
            else:
                return res,'未涉密',0
    except Exception as e:
        print(e)
        return [],'未涉密',1

def run_test(chain, test_path_root, output_res_path, output_error_path):
    """
    测试函数
    :param chain: 调用链
    :param test_path_root: 测试数据目录
    :param output_res_path: 输出结果文件路径
    :param output_error_path: 输出错误结果文件路径
    """
    # 读取训练集数据，转为dict
    test_paras = pd.read_excel(os.path.join(test_path_root, 'test_paras.xlsx'))
    test_files = pd.read_excel(os.path.join(test_path_root, 'test_files.xlsx'))

    test_paras = test_paras[['paragraph_id','content','points','paragraph_security_level','file_id']][:-3]
    test_files = test_files[['file_id','project_type']]
    test_df = pd.merge(test_paras,test_files, on='file_id', how='left')
    test_dict = test_df.to_dict(orient='records')
    
    # 自动建目录
    os.makedirs(os.path.dirname(output_res_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_error_path), exist_ok=True)

    # 提前写入文件头
    with open(output_res_path, 'w', encoding='utf-8') as f_res, \
         open(output_error_path, 'w', encoding='utf-8') as f_err:
        f_res.write("[\n")
        f_err.write("[\n")

        first_res = True
        first_err = True

        for para in tqdm(test_dict, desc='标注进度：', total=len(test_dict)):
            
            text = para['content']
            project_type = para['project_type']
            para['points'] = eval(para['points'])
            try:
                res = chain.call(text, project_type)
                points_llm, paragraph_security_level_llm, instruction_error = process_para(res)
            except Exception as e:
                print(e)
                points_llm= []
                paragraph_security_level_llm= '未涉密'
                instruction_error = 1
                res = 'llm error'

            para['points_llm'] = points_llm
            para['paragraph_security_level_llm'] = paragraph_security_level_llm
            para['instruction_error'] = instruction_error

            # 逐条写入结果文件
            if not first_res:
                f_res.write(",\n")
            json.dump(para, f_res, ensure_ascii=False, indent=4)
            first_res = False

            if instruction_error == 1:
                if not first_err:
                    f_err.write(",\n")
                json.dump(para, f_err, ensure_ascii=False, indent=4)
                first_err = False

        # 结束 JSON 数组
        f_res.write("\n]")
        f_err.write("\n]")
    
def evaluate(res_path, output_metrics_path, output_para_error_path):
    """
    评估函数
    :param res_path:
    :param output_metrics_path: 评估指标输出路径
    :param output_para_error_path: 段落错误输出路径
    """
    with open(res_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    
    security_level = []
    security_level_llm = []

    instruction_error_list = []

    points_list = []
    points_llm_list = []
    para_error_res = []
    
    for res in data:
        # 取段落层级的密级
        security_level.append(res['paragraph_security_level'])
        security_level_llm.append(res['paragraph_security_level_llm'] if res['paragraph_security_level_llm'] in ['一级','二级','三级','未涉密'] else '未涉密')
        # 是否出现指令遵循错误
        instruction_error_list.append(res['instruction_error'] if res['paragraph_security_level_llm'] != [] else 1)
        # 判断密点是否标注正确
        points_list.append(res['points'])
        points_llm_list.append(res['points_llm'])

        if res['paragraph_security_level'] != res['paragraph_security_level_llm']:
            para_error_res.append(res)

    # 自动建目录
    os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_para_error_path), exist_ok=True)

    # 段落密级效果测评
    y_true,y_pred = process_security_level(security_level,security_level_llm)
    para_metrics = calculate_para_metrics(y_true,y_pred)
    metrics_to_file(para_metrics,output_metrics_path)

    # 指令遵循错误
    instruction_metrics = instruction_error_rate(instruction_error_list)
    metrics_to_file(instruction_metrics,output_metrics_path)

    # 密点标注效果测评
    points_metrics = calculate_points_metrics2(points_list,points_llm_list)
    metrics_to_file(points_metrics,output_metrics_path)

    with open(output_para_error_path,'w') as f:
        json.dump(para_error_res,f,ensure_ascii=False,indent=4)

    print("指标评估已完毕")


# ---------------- 主入口 ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["direct","rag","rule","combine"],
                        help="测试模式: direct / rag / rule / combine")
    parser.add_argument("--test_path_root", type=str, default="dataset/test/",
                        help="测试集路径 (包含 test_paras.xlsx 和 test_files.xlsx)")
    parser.add_argument("--output_res_path", type=str, default="results/res.json",
                        help="模型输出结果文件")
    parser.add_argument("--output_error_path", type=str, default="results/error.json",
                        help="模型错误结果文件")
    parser.add_argument("--output_metrics_path", type=str, default="results/res.txt",
                        help="评估结果文件")
    parser.add_argument("--output_para_error_path", type=str, default="results/para_error.json")

    args = parser.parse_args()

    # 根据 mode 选择链
    if args.mode == "direct":
        chain = ParaLabelChain()
    elif args.mode == "rag":
        chain = RAGChain(rag_config)
    elif args.mode == "rule":
        chain = RULEChain(rule_config)
    elif args.mode == "combine":
        chain = CombineChain(combine_example_config, combine_rule_config)

    run_test(chain, args.test_path_root, args.output_res_path, args.output_error_path)

    merge_json("combine_test_res5.json",args.output_res_path,args.output_res_path)

    mark_instruction_error(args.output_res_path, args.output_res_path)

    evaluate(args.output_res_path, args.output_metrics_path, args.output_para_error_path)
