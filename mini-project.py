from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import nltk, re, torch
from collections import Counter

nltk.download('punkt', quiet=True)

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=-1)

def preprocess(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s!?.,\'\"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sentence_list(text):
    try:
        sents = nltk.tokenize.sent_tokenize(text)
    except LookupError:
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    return sents

def chunk_by_token_limit(text, max_tokens=None):
    if max_tokens is None:
        max_tokens = tokenizer.model_max_length - 64
    sents = sentence_list(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        s_ids = tokenizer.encode(s, add_special_tokens=False)
        s_len = len(s_ids)
        if s_len > max_tokens:
            # sentence too long — split by words into sub-sentences
            words = s.split()
            part = []
            part_len = 0
            for w in words:
                w_ids = tokenizer.encode(w + " ", add_special_tokens=False)
                wl = len(w_ids)
                if part_len + wl > max_tokens:
                    chunks.append(" ".join(part))
                    part = [w]
                    part_len = len(tokenizer.encode(w + " ", add_special_tokens=False))
                else:
                    part.append(w)
                    part_len += wl
            if part:
                # try to append to current if fits
                if cur_len + part_len <= max_tokens:
                    cur.append(" ".join(part))
                    cur_len += part_len
                else:
                    if cur:
                        chunks.append(" ".join(cur))
                    chunks.append(" ".join(part))
                    cur = []
                    cur_len = 0
        else:
            if cur_len + s_len <= max_tokens:
                cur.append(s)
                cur_len += s_len
            else:
                if cur:
                    chunks.append(" ".join(cur))
                cur = [s]
                cur_len = s_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def pegasus_chunk_summarize(text, max_tokens_per_chunk=None):
    chunks = chunk_by_token_limit(text, max_tokens=max_tokens_per_chunk)
    summaries = []
    for c in chunks:
        inputs = tokenizer(c, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_length=min(300, max(40, int(inputs['input_ids'].shape[1] * 0.45))),
                min_length=max(12, int(inputs['input_ids'].shape[1] * 0.12)),
                num_beams=8,
                length_penalty=1.1,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        summary = tokenizer.decode(out[0], skip_special_tokens=True)
        summaries.append(summary.strip())
    combined = " ".join(summaries).strip()
    if len(combined.split()) < 30:
        return combined
    # final compression pass
    inputs = tokenizer(combined, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_length=min(400, max(80, int(inputs['input_ids'].shape[1] * 0.45))),
            min_length=max(25, int(inputs['input_ids'].shape[1] * 0.12)),
            num_beams=10,
            length_penalty=1.2,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
    final = tokenizer.decode(out[0], skip_special_tokens=True)
    return final.strip()

def aggregate_emotion(text, max_tokens_per_chunk=None):
    if max_tokens_per_chunk is None:
        max_tokens_per_chunk = min(256, tokenizer.model_max_length // 2)
    chunks = chunk_by_token_limit(text, max_tokens=max_tokens_per_chunk)
    if not chunks:
        return "neutral", 0.0
    votes = Counter()
    confs = []
    for c in chunks:
        res = emotion_model(c)
        if isinstance(res, list) and len(res) > 0:
            r = res[0]
        else:   
            r = res
        label = r['label']
        score = float(r.get('score', 0.0))
        votes[label] += 1
        confs.append((label, score))
    top_label = votes.most_common(1)[0][0]
    # average confidence for top_label across chunks that predicted it
    selected = [s for (lab, s) in confs if lab == top_label]
    avg_conf = (sum(selected)/len(selected)) if selected else 0.0
    return top_label, float(avg_conf)

def analyze_text(text):
    text = preprocess(text)
    emotion_label, emotion_conf = aggregate_emotion(text, max_tokens_per_chunk=256)
    summary = pegasus_chunk_summarize(text, max_tokens_per_chunk=tokenizer.model_max_length - 64)
    emotion_map = {"joy":"happy","anger":"angry","sadness":"sad","fear":"afraid","surprise":"surprised","disgust":"disgusted","neutral":"neutral"}
    emotion_word = emotion_map.get(emotion_label.lower(), emotion_label.lower())
    intensity = "extremely " if emotion_conf > 0.85 else ("very" if emotion_conf > 0.7 else "slightly")
    interpretation = f"The person is {intensity} {emotion_word} about {summary[0].lower() + summary[1:] if summary else summary}."
    print("\n===============================")
    print(f"🧩 Original Text: {text}")
    print(f"🎭 Detected Emotion: {emotion_label} ({emotion_conf:.2f})")
    print(f"📝 Summary: {summary}")
    print(f"💬 Final Interpretation: {interpretation}")
    print("===============================")

text = "India's growth story is a compelling narrative of economic resurgence, social transformation, and technological prowess, marking its ascent as a prominent global player. The nation's economy, one of the fastest-growing in the world, is projected to become the third-largest by 2030, fueled by a robust services sector, a burgeoning manufacturing base, and significant government spending on infrastructure. This economic dynamism is complemented by remarkable social progress, with millions lifted out of poverty, a steady increase in life expectancy and literacy rates, and a growing middle class that is reshaping consumer markets. The digital revolution has been a cornerstone of this transformation, with initiatives like Digital India fostering financial inclusion, improving governance, and creating a vibrant startup ecosystem that is driving innovation in areas like fintech and e-commerce. Technological advancements extend to space exploration, where India has achieved significant milestones with missions to the Moon and Mars, showcasing its scientific and engineering capabilities. A massive infrastructure push is further catalyzing growth, with the development of modern highways, high-speed rail networks, and world-class airports enhancing connectivity and facilitating trade. In the international arena, India's influence continues to expand, guided by a foreign policy of strategic autonomy and multi-alignment. As a leading voice in global forums and a key player in regional security, India is actively shaping the discourse on issues ranging from climate change to counter-terrorism. This multifaceted growth, however, is not without its challenges, including the need to address income inequality, create sufficient employment opportunities for its youthful population, and navigate a complex geopolitical landscape. Nevertheless, with its demographic dividend, democratic values, and a clear vision for the future, India is poised to continue its remarkable journey of growth and development, solidifying its position as a major force in the 21st century."
analyze_text(text)
 

