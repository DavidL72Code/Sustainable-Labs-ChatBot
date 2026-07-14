from pathlib import Path
from run_questions_eval import load_dotenv_simple, load_chatbot_symbols
import run_questions_eval as R
load_dotenv_simple(Path('.env'))
load_chatbot_symbols()
import Chatbot
bot = Chatbot.create_chatbot(R.ChatbotConfig())

orig_rc = bot.retrieve_context
def spy_rc(query, top_k=None, query_route=None):
    ctx, meta, diag = orig_rc(query, top_k, query_route)
    print(f"  [retrieve top-{len(ctx)}]:")
    for i,m in enumerate(meta[:6],1):
        sp=(m or {}).get('source_path','?').split('/')[-1]
        sec=(m or {}).get('section_name','')[:35]
        cl=(m or {}).get('chunk_level','')
        print(f"    [{i}] {sp} | {cl} | {sec}")
    return ctx, meta, diag
bot.retrieve_context = spy_rc

cases = [
    ("fs_004", "Who does SSL say its transdisciplinary research centers and is led by?", []),
    ("fs_006", "What does Sarah Mayorga, Associate Professor at Brandeis University, say she values about her work with SSL and the Northeast Climate Justice Research Collaborative?", []),
    ("fs_009", "In what season and year was the Northeast Climate Justice Research Collaborative launched?", []),
]
for cid, q, hist in cases:
    print("="*70); print(cid, q[:60])
    rt = bot.detect_local_query_route(q)
    print(f"  route={rt.get('question_type')} | entity_reg={bot.should_use_entity_registry(q,rt)} | sect_reg={bot.should_use_section_registry(q,rt)}")
    r = bot.answer(q, recent_history=[])
    print(f"  reply: {r['reply'][:120]}")
