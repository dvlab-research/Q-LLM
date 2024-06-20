import torch

class GreedySearch:
    def __init__(self, model, tokenizer):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.past_kv = None
        self.device = self.model.device

    def clear(self):
        self.past_kv = None

    def _process_texts(self, input_text):
        model_inputs = {}
        input_ids = self.tokenizer.encode(input_text)

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(model_inputs["input_ids"])

        for key in model_inputs:
            model_inputs[key] = torch.tensor(model_inputs[key]).int().unsqueeze(0).cuda()

        return model_inputs


    def generate(self, input_ids=None, text=None, **kwargs):
        if input_ids is None and text is not None:
            model_inputs = self._process_texts(text)
            input_ids = model_inputs['input_ids']

        with torch.inference_mode():
            result = self._decode(input_ids, **kwargs)
        return result

    def _decode(
            self, input_ids, max_length=100, extra_end_token_ids=[], chunk_size=4096, output=False, inputs_embeds=None, images=None, image_sizes=None, attention_mask=None, use_cache=True, 
            return_dict=True, **kwargs):

        if images is not None:
            kwargs['question_ids'] = [0, input_ids.size(-1)]
            input_ids, _, _, _, inputs_embeds, _ = self.model.prepare_inputs_labels_for_multimodal(
                input_ids,     
                images=images,
                image_sizes=image_sizes,)

        if input_ids is not None:
            if input_ids.dim() == 1:
                input_ids = input_ids[None, :]
            input_ids = input_ids.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, device=self.device)
            assert input_ids.size(0) == 1
            length = input_ids.size(1)
        else:
            # print('input embeds:', inputs_embeds.shape)
            inputs_embeds = inputs_embeds.to(self.device)
            if attention_mask is None:
                attention_mask = torch.ones_like(inputs_embeds[:, :, 0], device=self.device)
            assert inputs_embeds.size(0) == 1
            length = inputs_embeds.size(1)
            generated_words = None

        end_token_ids = extra_end_token_ids + [self.tokenizer.eos_token_id]
        logits = None
        past_key_values = self.past_kv
        if output:
            output_text = ""
        for i in range(max_length + 1):
            if i == 0:
                if chunk_size is None:
                    chunk_size = length
                for st in range(0, length - 1, chunk_size):
                    ed = min(length - 1, st + chunk_size)
                    out = self.model(
                        input_ids = input_ids[:, st: ed] if input_ids is not None else None,
                        attention_mask = attention_mask[:, :ed],
                        inputs_embeds = inputs_embeds[:, st: ed] if inputs_embeds is not None else None,
                        use_cache = True,
                        return_dict = True,
                        past_key_values = past_key_values,
                        **kwargs
                    )
                    logits, past_key_values = out.logits, out.past_key_values
                    
                ###############################################################
                out = self.model(
                    input_ids = input_ids[:, -1:] if input_ids is not None else None,
                    attention_mask = attention_mask,
                    inputs_embeds = inputs_embeds[:, -1:] if inputs_embeds is not None else None,
                    use_cache = True,
                    return_dict = True,
                    past_key_values = past_key_values,
                    **kwargs
                )
                logits, past_key_values = out.logits, out.past_key_values

            else:
                ###############################################################
                out = self.model(
                    input_ids = input_ids[:, -1:] if input_ids is not None else None,
                    attention_mask = attention_mask,
                    inputs_embeds = inputs_embeds[:, -1:] if inputs_embeds is not None else None,
                    use_cache = True,
                    return_dict = True,
                    past_key_values = past_key_values,
                    **kwargs
                )
                logits, past_key_values = out.logits, out.past_key_values

            logits = logits[:, -1, :]
            word = logits.argmax(dim=-1) # [1]
            if word.item() in end_token_ids or i == max_length:
                break

            if input_ids is not None:
                input_ids = torch.cat((input_ids, word.view(1, 1)), dim=-1)
            else:
                embeds = self.model.get_input_embeddings() # Embedding(128256, 4096)
                word_embeds = embeds(word) # torch.Size([1, 4096])
                inputs_embeds = torch.cat((inputs_embeds, word_embeds.unsqueeze(0)), dim=1)
                if generated_words is None:
                    generated_words = word
                else:
                    generated_words = torch.cat((generated_words, word), dim=-1)  
                
            attention_mask = torch.cat(
                (attention_mask, torch.ones((attention_mask.size(0), 1), dtype=torch.int, device=attention_mask.device)),
                dim=-1
            )

            if output:
                tmp = self.tokenizer.decode(input_ids.squeeze(0)[length:])
                if len(tmp) > len(output_text):
                    import sys               
                    sys.stdout.write(tmp[len(output_text):])
                    sys.stdout.flush()
                    output_text = tmp

        self.past_kv = past_key_values

        if output:
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        if input_ids is not None:
            return [self.tokenizer.decode(input_ids.squeeze(0)[length:])]
        else:
            return [generated_words]
