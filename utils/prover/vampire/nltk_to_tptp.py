import json
from nltk.sem.logic import *

def _convert_to_tptp(expression):
    """
    Convert ``logic.Expression`` to TPTP formatted string.
    """
    if isinstance(expression, ExistsExpression):
        # 存在量词 ∃ -> ?[X]
        return (
            "?["
            + str(expression.variable).upper()  # TPTP 变量要求是大写
            + "] : "
            + _convert_to_tptp(expression.term)
        )
    elif isinstance(expression, AllExpression):
        # 全称量词 ∀ -> ![X]
        return (
            "!["
            + str(expression.variable).upper()
            + "] : "
            + _convert_to_tptp(expression.term)
        )
    elif isinstance(expression, NegatedExpression):
        # 否定 ¬ -> ~
        return "~(" + _convert_to_tptp(expression.term) + ")"
    elif isinstance(expression, AndExpression):
        # 合取 ∧ -> &
        return (
            "("
            + _convert_to_tptp(expression.first)
            + " & "
            + _convert_to_tptp(expression.second)
            + ")"
        )
    elif isinstance(expression, OrExpression):
        # 析取 ∨ -> |
        return (
            "("
            + _convert_to_tptp(expression.first)
            + " | "
            + _convert_to_tptp(expression.second)
            + ")"
        )
    elif isinstance(expression, ImpExpression):
        # 蕴含 → -> =>
        return (
            "("
            + _convert_to_tptp(expression.first)
            + " => "
            + _convert_to_tptp(expression.second)
            + ")"
        )
    elif isinstance(expression, IffExpression):
        # 双条件 ↔ -> <=>
        return (
            "("
            + _convert_to_tptp(expression.first)
            + " <=> "
            + _convert_to_tptp(expression.second)
            + ")"
        )
    elif isinstance(expression, EqualityExpression):
        # 等式 = -> =
        return (
            "("
            + _convert_to_tptp(expression.first)
            + " = "
            + _convert_to_tptp(expression.second)
            + ")"
        )
    elif isinstance(expression, ApplicationExpression):
        # 处理谓词应用表达式：只将谓词名的第一个字母转换为小写，参数转换为大写
        predicate = str(expression.function)
        predicate = predicate[0].lower() + predicate[1:]  # 只将第一个字母小写
        terms = ', '.join(_convert_to_tptp(term) for term in expression.args)  # 参数转换
        return f"{predicate}({terms})"

    elif isinstance(expression, Variable):
        # 如果变量是小写的 x 等，则转换为大写
        return str(expression).upper()
    else:
        # 常量处理，确保常量为小写（如 Rina -> rina）
        return str(expression).lower()
 
#############
  
# # 示例表达式
# exp = Expression.fromstring("(JokesAboutAddiction(Rina) | UnawareOfCaffeineDrug(Rina))")

# # 调用转换函数
# result = _convert_to_tptp(exp)

# # 输出结果
# print(result)

####################

# 读取 JSONL 文件
def read_jsonl_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

# 写入 JSONL 文件
def write_jsonl_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry) + '\n')

# 转换 FOL 表达式为 TPTP
def convert_nltk_to_tptp(data):
    for entry in data:
        # 转换 conclusion-FOL
        fol_conclusion = Expression.fromstring(entry['conclusion-FOL'])
        entry['conclusion-FOL'] = _convert_to_tptp(fol_conclusion)  # 覆盖原始字段
        print(f"Converted conclusion: {entry['conclusion-FOL']}")  # 调试信息
        
        # 转换 premises-FOL
        tptp_premises = []
        for premise in entry['premises-FOL']:
            fol_premise = Expression.fromstring(premise)
            converted_premise = _convert_to_tptp(fol_premise)
            tptp_premises.append(converted_premise)
            print(f"Converted premise: {converted_premise}")  # 调试信息
        entry['premises-FOL'] = tptp_premises  # 覆盖原始字段
    return data

# 主函数
def main():
    # 读取原始 JSONL 文件
    input_filename = 'folio-train.jsonl'
    data = read_jsonl_file(input_filename)
    
    # 转换 FOL 为 TPTP
    tptp_data = convert_nltk_to_tptp(data)
    
    # 将新数据写入新的 JSONL 文件
    output_filename = 'folio-train-tptp.jsonl'
    write_jsonl_file(tptp_data, output_filename)
    
    print(f"TPTP 格式的文件已保存为 {output_filename}")

# 执行主函数
if __name__ == "__main__":
    main()



#################

# from nltk.sem.logic import *

# def _convert_to_tptp(expression):
#     """
#     Convert ``logic.Expression`` to TPTP formatted string
#     """
#     if isinstance(expression, ExistsExpression):
#         # 存在量词 ∃ -> ?[X]
#         return (
#             "?["
#             + str(expression.variable).upper()  # TPTP 变量要求是大写
#             + "] : "
#             + _convert_to_tptp(expression.term)
#         )
#     elif isinstance(expression, AllExpression):
#         # 全称量词 ∀ -> ![X]
#         return (
#             "!["
#             + str(expression.variable).upper()
#             + "] : "
#             + _convert_to_tptp(expression.term)
#         )
#     elif isinstance(expression, NegatedExpression):
#         # 否定 ¬ -> ~
#         return "~(" + _convert_to_tptp(expression.term) + ")"
#     elif isinstance(expression, AndExpression):
#         # 合取 ∧ -> &
#         return (
#             "("
#             + _convert_to_tptp(expression.first)
#             + " & "
#             + _convert_to_tptp(expression.second)
#             + ")"
#         )
#     elif isinstance(expression, OrExpression):
#         # 析取 ∨ -> |
#         return (
#             "("
#             + _convert_to_tptp(expression.first)
#             + " | "
#             + _convert_to_tptp(expression.second)
#             + ")"
#         )
#     elif isinstance(expression, ImpExpression):
#         # 蕴含 → -> =>
#         return (
#             "("
#             + _convert_to_tptp(expression.first)
#             + " => "
#             + _convert_to_tptp(expression.second)
#             + ")"
#         )
#     elif isinstance(expression, IffExpression):
#         # 双条件 ↔ -> <=>
#         return (
#             "("
#             + _convert_to_tptp(expression.first)
#             + " <=> "
#             + _convert_to_tptp(expression.second)
#             + ")"
#         )
#     elif isinstance(expression, EqualityExpression):
#         # 等式 = -> =
#         return (
#             "("
#             + _convert_to_tptp(expression.first)
#             + " = "
#             + _convert_to_tptp(expression.second)
#             + ")"
#         )
#     elif isinstance(expression, ApplicationExpression):
#         # 处理谓词应用表达式：谓词名转换为小写且不使用下划线，参数转换为大写
#         predicate = ''.join(str(expression.function).split()).lower()  # 将谓词名中的所有单词小写并连写
#         terms = ', '.join(_convert_to_tptp(term) for term in expression.args)  # 参数转换
#         return f"{predicate}({terms})"
#     elif isinstance(expression, Variable):
#         # 变量转换为大写
#         return str(expression).upper()
#     else:
#         # 默认情况下，返回字符串表示
#         return str(expression)

# # 示例表达式
# exp = Expression.fromstring("all x. (P(x) -> Q(x))")

# # 调用转换函数
# result = _convert_to_tptp(exp)

# # 输出结果
# print(result)




