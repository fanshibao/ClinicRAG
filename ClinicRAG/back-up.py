def inference(self, query, history=[], candidate_diseases=None):
    self.history = history
    prompt = self.generate_prompt(query, self.history)
    # print(prompt,flush=True)
    generated_text = self.model_genrate(prompt)
    cur_generate = generated_text

    # For diagnosis.
    if 'Enter the diagnostic process, analyzing patient symptoms:' in generated_text or 'Analyzing patient symptoms:' in generated_text or '进入诊断流程，分析病人症状信息:' in generated_text or '总结病人症状信息:' in generated_text:
        if 'Enter the diagnostic process, analyzing patient symptoms:' in generated_text or 'Analyzing patient symptoms:' in generated_text:
            is_en = True
        else:
            is_en = False

        p_syms = re.findall(r'(?:\n|,)\s*"([^"]*?)\s*"', generated_text)
        f_syms = re.findall(r'(?:没有|No)\s*"([^"]*?)\s*"', generated_text)

        self.sym_info = {'true_syms': [], 'false_syms': []}
        for sym in p_syms:
            if sym not in self.sym_info['true_syms']:
                self.sym_info['true_syms'].append(sym)
        for sym in f_syms:
            if sym not in self.sym_info['false_syms']:
                self.sym_info['false_syms'].append(sym)

        if is_en:
            if not candidate_diseases:
                candi = self.retriever_en.find_top_k(self.sym_info['true_syms'], self.sym_info['false_syms'])
            else:
                candi = self.retriever_en.find_top_k_with_candis(self.sym_info['true_syms'],
                                                                 self.sym_info['false_syms'], candidate_diseases)
        else:
            if not candidate_diseases:
                candi = self.retriever_zh.find_top_k(self.sym_info['true_syms'], self.sym_info['false_syms'])
            else:
                candi = self.retriever_zh.find_top_k_with_candis(self.sym_info['true_syms'],
                                                                 self.sym_info['false_syms'], candidate_diseases)

        t2_str = self.get_candidate_dis(candi, is_en)

        cur_generate += t2_str
        generated_text = self.model_genrate(prompt + cur_generate)

        try:
            if is_en:
                pattern = r"## Diagnostic confidence:\s*(.*)"
            else:
                pattern = r"## 诊断置信度:\s*(.*)"
            match = re.search(pattern, generated_text, re.DOTALL)
            confidence_distribution = json.loads(match.group(1))
        except Exception as e:
            print('Error')
            print(e)
            print(confidence_distribution)
            return cur_generate + generated_text

        cur_generate += generated_text
        max_key = max(confidence_distribution, key=confidence_distribution.get)
        max_value = confidence_distribution[max_key]

        if max_value > self.confidence_threshold:
            # diagnosis
            if is_en:
                t4_1 = '\n\n## Diagnosis:\n'
            else:
                t4_1 = '\n\n## 做出诊断:\n'
            # print(t4_1,end='',flush = True)
            cur_generate += t4_1
            generated_text = self.model_genrate(prompt + cur_generate)
            cur_generate += generated_text

            self.history = self.history + [(query, cur_generate)]

        else:
            # Diagnosis
            if is_en:
                t4_2 = '\n\n\nInadequate for diagnosis. Ask for symptoms:\n'
            else:
                t4_2 = '\n\n\n为了更好的诊断疾病，我还需要了解您更多信息，请您回答我的问题:\n'
            # print(t4_2,end='',flush = True)
            cur_generate += t4_2
            generated_text = self.model_genrate(prompt + cur_generate)
            cur_generate += generated_text

            self.history = self.history + [(query, cur_generate)]

        return cur_generate, self.history, confidence_distribution
    # For other chat.
    else:
        self.history = self.history + [(query, cur_generate)]
        return cur_generate, self.history, {}