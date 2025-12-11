
'''
credit to original author: Glenn (chenluda01@outlook.com)
Author: Doragd
'''

import re
import os
import requests
import time
import json
import datetime
from io import BytesIO 
from tqdm import tqdm
from translate import translate
from pypdf import PdfReader


SERVERCHAN_API_KEY = os.environ.get("SERVERCHAN_API_KEY", None)
QUERY = os.environ.get('QUERY', 'cs.IR')
LIMITS = int(os.environ.get('LIMITS', 3))
# 支持多个飞书URL，使用逗号分隔
FEISHU_URLS = os.environ.get("FEISHU_URL", "").split(',')
# 去除空字符串和空格
FEISHU_URLS = [url.strip() for url in FEISHU_URLS if url.strip()]
MODEL_TYPE = os.environ.get("MODEL_TYPE", "DeepSeek")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

def get_yesterday():
    today = datetime.datetime.now()
    yesterday = today - datetime.timedelta(days=1)
    return yesterday.strftime('%Y-%m-%d')


def search_arxiv_papers(search_term, max_results=10):
    papers = []

    url = f'http://export.arxiv.org/api/query?' + \
          f'search_query=all:{search_term}' +  \
          f'&start=0&&max_results={max_results}' + \
          f'&sortBy=submittedDate&sortOrder=descending'

    response = requests.get(url)

    if response.status_code != 200:
        return []

    feed = response.text
    entries = feed.split('<entry>')[1:]

    if not entries:
        return []

    print('[+] 开始处理每日最新论文....')

    for entry in entries:

        title = entry.split('<title>')[1].split('</title>')[0].strip()
        summary = entry.split('<summary>')[1].split('</summary>')[0].strip().replace('\n', ' ').replace('\r', '')
        url = entry.split('<id>')[1].split('</id>')[0].strip()
        pub_date = entry.split('<published>')[1].split('</published>')[0]
        pub_date = datetime.datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d")

        first_page_text = get_pdf_first_page_text(url)
        affiliation = extract_affiliation_with_deepseek(first_page_text)

        papers.append({
            'title': title,
            'affiliation': affiliation,
            'url': url,
            'pub_date': pub_date,
            'summary': summary,
            'translated': '',          # 摘要中文
            'title_translated': '',    # 标题中文
        })
    
    print('[+] 开始翻译每日最新论文并缓存....')

    papers = save_and_translate(papers)
    
    return papers

def get_pdf_first_page_text(abs_url, timeout=20):
    """
    通过 abs 链接构造 pdf 链接，下载并抽取第一页文本。
    """
    # 典型 abs 链接: https://arxiv.org/abs/2512.04847v1
    if "/abs/" not in abs_url:
        return ""

    pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

    try:
        resp = requests.get(pdf_url, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        print(f"[-] 下载 PDF 失败: {pdf_url}, err={e}")
        return ""

    try:
        reader = PdfReader(BytesIO(resp.content))
        if not reader.pages:
            return ""
        text = reader.pages[0].extract_text() or ""
        return text.strip()
    except Exception as e:
        print(f"[-] 解析 PDF 失败: {pdf_url}, err={e}")
        return ""


def extract_affiliation_with_deepseek(first_page_text):
    """
    调用 DeepSeek，从第一页文本中抽取作者单位。
    返回一个简短的字符串。
    """
    if not DEEPSEEK_API_KEY:
        print("⚠️ 未设置 DEEPSEEK_API_KEY，跳过单位抽取")
        return ""

    if not first_page_text.strip():
        return ""

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = (
        "下面是一篇 arXiv 论文 PDF 首页的文字内容。"
        "请从中提取所有作者的所属单位信息（学校、公司、研究机构等），"
        "如果有多个单位，请用中文输出，使用中文顿号分隔。"
        "不要输出任何解释或多余文字，只输出单位列表。"
        "\n\n论文首页内容：\n"
        f"{first_page_text}"
    )

    data = {
        "model": "deepseek-chat",  # 视你账号实际模型名而定
        "messages": [
            {"role": "system", "content": "你是一个信息抽取助手，只输出作者单位列表。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"].strip()
        content = dedup_affiliation_text(content)
        return content
    except Exception as e:
        print(f"[-] DeepSeek 抽取单位失败: {e}")
        return ""


def dedup_affiliation_text(text: str) -> str:
    """
    对 DeepSeek 返回的单位字符串做去重和清洗：
    - 按 顿号/逗号/分号/换行 切分
    - 去掉空字符串
    - 去重但保留原顺序
    """
    if not text:
        return ""
    parts = re.split(r"[、,;；\n]+", text)
    seen = set()
    result = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return "、".join(result)



def send_wechat_message(title, content, SERVERCHAN_API_KEY):
    url = f'https://sctapi.ftqq.com/{SERVERCHAN_API_KEY}.send'
    params = {
        'title': title,
        'desp': content,
    }
    requests.post(url, params=params)

def send_feishu_message(title, content, urls=None):
    # 如果没有指定URL列表，使用默认的FEISHU_URLS
    if urls is None:
        urls = FEISHU_URLS
    
    # 如果没有有效的飞书URL，直接返回
    if not urls:
        print("⚠️ 没有有效的飞书URL，跳过发送消息")
        return
    
    card_data = {
        "config": {
            "wide_screen_mode": True
        },
        "header": {
            "template": "green",
            "title": {
            "tag": "plain_text",
            "content": title
            }
        },
        "elements": [
            {
            "tag": "img",
            "img_key": "img_v2_9781afeb-279d-4a05-8736-1dff05e19dbg",
            "alt": {
                "tag": "plain_text",
                "content": ""
            },
            "mode": "fit_horizontal",
            "preview": True
            },
            {
            "tag": "markdown",
            "content": content
            }
        ]
    }
    card = json.dumps(card_data)
    body =json.dumps({"msg_type": "interactive","card":card})
    headers = {"Content-Type":"application/json"}
    
    # 向每个飞书URL发送消息
    for idx, url in enumerate(urls):
        try:
            requests.post(url=url, data=body, headers=headers, timeout=10)
            print(f"✉️ 飞书推送[{idx+1}/{len(urls)}]已发送")
        except Exception as e:
            print(f"❌ 飞书推送[{idx+1}/{len(urls)}]失败: {e}")


def save_and_translate(papers, filename='arxiv.json'):
    # 1. 读缓存
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        results = []

    cached_title2idx = {result['title'].lower(): i for i, result in enumerate(results)}

    untranslated_papers = []
    hit_cnt = 0

    for paper in papers:
        title_lower = paper['title'].lower()
        if title_lower in cached_title2idx:
            # 命中缓存 → 回填翻译/单位
            hit_cnt += 1
            cached = results[cached_title2idx[title_lower]]

            if 'translated' in cached and not paper.get('translated'):
                paper['translated'] = cached.get('translated', '')

            if 'title_translated' in cached and not paper.get('title_translated'):
                paper['title_translated'] = cached.get('title_translated', '')

            # 单位：优先用本次 DeepSeek 的，如果空再用缓存
            if not paper.get('affiliation') and cached.get('affiliation'):
                paper['affiliation'] = cached.get('affiliation', '')
        else:
            untranslated_papers.append(paper)

    # 2. 对“新论文”做摘要翻译
    summaries = [p['summary'] for p in untranslated_papers]
    if summaries:
        target_summaries = translate(summaries)
        if len(target_summaries) == len(untranslated_papers):
            for p, t in zip(untranslated_papers, target_summaries):
                p['translated'] = t

    # 3. 对“新论文”做标题翻译
    titles = [p['title'] for p in untranslated_papers]
    if titles:
        target_titles = translate(titles)
        if len(target_titles) == len(untranslated_papers):
            for p, t in zip(untranslated_papers, target_titles):
                p['title_translated'] = t

    # 4. 写回缓存
    results.extend(untranslated_papers)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f'[+] 总检索条数: {len(papers)} | 命中缓存: {hit_cnt} | 实际翻译: {len(untranslated_papers)}....')

    return papers

        
def cronjob():

    if SERVERCHAN_API_KEY is None:
        raise Exception("未设置SERVERCHAN_API_KEY环境变量")

    print('[+] 开始执行每日推送任务....')

    yesterday = get_yesterday()
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    print('[+] 开始检索每日最新论文....')
    papers = search_arxiv_papers(QUERY, LIMITS)

    if papers == []:
        
        push_title = f'Arxiv:{QUERY}[X]@{today}'
        send_wechat_message('', '[WARN] NO UPDATE TODAY!', SERVERCHAN_API_KEY)

        print('[+] 每日推送任务执行结束')

        return True
        

    print('[+] 开始推送每日最新论文....')

    for ii, paper in enumerate(tqdm(papers, total=len(papers), desc=f"论文推送进度")):

        title = paper['title']
        url = paper['url']
        pub_date = paper['pub_date']
        summary = paper['summary']
        translated = paper['translated']
        affiliation = paper.get("affiliation", "").strip()

        yesterday = get_yesterday()

        if pub_date == yesterday:
            msg_title = f'[Newest]{title}' 
        else:
            msg_title = f'{title}'

        title_cn = paper.get("title_translated", "").strip()

        msg_url = f'URL: {url}'
        msg_pub_date = f'Pub Date：{pub_date}'
        # 中文摘要
        msg_translated = f'Translated Summary (Powered by {MODEL_TYPE}):\n\n{translated}'

        msg_affiliation = ""
        if affiliation:
            msg_affiliation = f'Affiliation（DeepSeek 抽取）：\n{affiliation}'

        msg_title_cn = ""
        if title_cn:
            msg_title_cn = f'Title（中文）：{title_cn}'

        push_title = f'Arxiv:{QUERY}[{ii}]@{today}'
        msg_content = f"[{msg_title}]({url})\n\n{msg_pub_date}\n\n"

        # 先中文标题
        if msg_title_cn:
            msg_content += msg_title_cn + "\n\n"

        # 再作者单位
        if msg_affiliation:
            msg_content += msg_affiliation + "\n\n"

        # 链接 + 中文摘要
        msg_content += f"{msg_url}\n\n{msg_translated}\n\n"

        # send_wechat_message(push_title, msg_content, SERVERCHAN_API_KEY)
        send_feishu_message(push_title, msg_content)

        time.sleep(12)

    print('[+] 每日推送任务执行结束')

    return True


if __name__ == '__main__':
    cronjob()



