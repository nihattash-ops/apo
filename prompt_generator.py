from openai import OpenAI

class PromptGenerator:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None, base_url=None):
        if not api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate_candidates(self, base_prompt, num_candidates=3, feedback=None):
        """
        根据基础提示词和可选的反馈，生成多个候选提示词。
        """
        candidate_prompts = []
        instruction = f"You are an expert prompt engineer. Your goal is to generate {num_candidates} variations of the following prompt to improve its performance for a given task. The variations should be distinct and explore different phrasing, emphasis, or additional instructions. Each variation should be on a new line, prefixed with 'PROMPT:'.\n\nOriginal Prompt: {base_prompt}"
        
        if feedback:
            instruction += f"\n\nConsider the following feedback: {feedback}"

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": instruction}
                ],
                temperature=0.7, # Use higher temperature for creativity
                n=1 # Request one response that contains multiple candidates
            )
            generated_text = response.choices[0].message.content.strip()
            
            # Parse the generated text to extract candidate prompts
            for line in generated_text.split('\n'):
                if line.startswith('PROMPT:'):
                    candidate_prompts.append(line[len('PROMPT:'):].strip())
            
            # Ensure we have at least num_candidates, if LLM didn't generate enough
            while len(candidate_prompts) < num_candidates:
                # Fallback: just re-use the base prompt or generate a simple variation
                candidate_prompts.append(f"Rephrased: {base_prompt}") # Simple fallback
                
            return candidate_prompts[:num_candidates]

        except Exception as e:
            print(f"Error generating candidate prompts: {e}")
            return [base_prompt] * num_candidates
