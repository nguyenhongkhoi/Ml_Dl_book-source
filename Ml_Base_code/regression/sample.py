"""
Công cụ khám phá link & API endpoint công khai trên một trang web.
Tối ưu để ưu tiên các endpoint liên quan tới "mã lớp", "giáo viên", "lịch học"...

LƯU Ý: Chỉ chạy trên hệ thống bạn có quyền kiểm tra. Tool này KHÔNG khai thác
lỗ hổng, KHÔNG bypass đăng nhập — chỉ đọc nội dung HTML/JS công khai và
phân tích bằng regex.
"""

import re
import time
import urllib.parse
from collections import deque

import requests
from bs4 import BeautifulSoup

# ====================== CẤU HÌNH ======================
TARGET_URL = "https://e.hust.edu.vn"
MAX_DEPTH = 2              # độ sâu crawl (0 = chỉ trang chính)
MAX_PAGES = 60             # giới hạn số trang HTML sẽ quét (tránh quét vô hạn)
REQUEST_DELAY = 0.5        # giây nghỉ giữa các request, tránh spam server
TIMEOUT = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# Từ khóa ưu tiên: mã lớp, giáo viên, lịch học...
PRIORITY_KEYWORDS = [
    "lop", "class", "malop", "classid", "class_id",
    "giangvien", "giaovien", "teacher", "lecturer", "instructor", "gv",
    "monhoc", "subject", "course",
    "tkb", "schedule", "timetable", "lichhoc",
    "sinhvien", "student", "mssv",
    "diem", "grade", "score",
]

# ====================== REGEX ======================
# Bắt các path dạng /api/..., /v1/..., /ajax/..., /rest/...
API_PATH_PATTERN = re.compile(
    r"""["'`]?(/(?:api|v[0-9]+|ajax|endpoint|rest|service|json|graphql)[a-zA-Z0-9_\-/]*)["'`]?""",
    re.IGNORECASE,
)

# Bắt các URL tuyệt đối có path giống API (trong file JS thường gọi full URL)
ABS_API_URL_PATTERN = re.compile(
    r"""https?://[a-zA-Z0-9_.\-]+/(?:api|v[0-9]+|ajax|endpoint|rest|service)[a-zA-Z0-9_\-/]*""",
    re.IGNORECASE,
)

# Bắt fetch("..."), axios.get("..."), $.ajax({url: "..."}) trong JS
JS_CALL_PATTERN = re.compile(
    r"""(?:fetch|axios\.(?:get|post|put|delete)|url\s*:|\.ajax)\s*\(?\s*["'`]([^"'`]+)["'`]""",
    re.IGNORECASE,
)

# Bắt các key JSON gợi ý dữ liệu lớp/giáo viên, ví dụ: "MaLop":, "TenGV":, "GiangVien":
JSON_KEY_PATTERN = re.compile(
    r'["\']([a-zA-Z_]{2,30})["\']\s*:', re.IGNORECASE
)

# ====================== STATE ======================
visited_pages = set()
found_urls = set()
found_apis = set()
priority_apis = set()       # API có khả năng liên quan mã lớp / giáo viên
interesting_json_keys = set()

session = requests.Session()
session.headers.update(HEADERS)


def is_same_domain(url, target_netloc):
    return urllib.parse.urlparse(url).netloc == target_netloc


def matches_priority(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in PRIORITY_KEYWORDS)


def fetch(url):
    try:
        resp = session.get(url, timeout=TIMEOUT)
        time.sleep(REQUEST_DELAY)
        if resp.status_code == 200:
            return resp
    except Exception as e:
        print(f"[!] Lỗi khi tải {url}: {e}")
    return None


def extract_apis_from_text(text, base_url):
    """Tìm API endpoint trong nội dung text (HTML hoặc JS)."""
    candidates = set()

    for m in API_PATH_PATTERN.findall(text):
        candidates.add(urllib.parse.urljoin(base_url, m))

    for m in ABS_API_URL_PATTERN.findall(text):
        candidates.add(m)

    for m in JS_CALL_PATTERN.findall(text):
        # Bỏ qua data: hoặc URL không hợp lệ
        if m.startswith(("http", "/")):
            candidates.add(urllib.parse.urljoin(base_url, m))

    for c in candidates:
        found_apis.add(c)
        if matches_priority(c):
            priority_apis.add(c)

    # Quét key JSON gợi ý dữ liệu lớp/giáo viên (chỉ lấy key trùng từ khóa ưu tiên)
    for key in JSON_KEY_PATTERN.findall(text):
        if matches_priority(key):
            interesting_json_keys.add(key)


def extract_links_and_apis(url, target_netloc, queue, depth):
    if url in visited_pages or len(visited_pages) >= MAX_PAGES:
        return
    visited_pages.add(url)

    resp = fetch(url)
    if resp is None:
        return

    content_type = resp.headers.get("Content-Type", "")

    # Nếu là file JS, chỉ cần regex trên nội dung, không parse HTML
    if "javascript" in content_type or url.endswith(".js"):
        extract_apis_from_text(resp.text, url)
        return

    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. Link & resource trong HTML
    for tag in soup.find_all(["a", "link", "script", "img", "iframe"]):
        link = tag.get("href") or tag.get("src")
        if not link:
            continue
        full_url = urllib.parse.urljoin(url, link)

        if is_same_domain(full_url, target_netloc):
            if full_url not in found_urls:
                found_urls.add(full_url)
            # Đưa file JS / trang HTML vào queue để quét sâu hơn
            if depth < MAX_DEPTH:
                if full_url.endswith(".js") or "." not in full_url.rsplit("/", 1)[-1]:
                    queue.append((full_url, depth + 1))

    # 2. Tìm API ngay trong HTML/inline script
    extract_apis_from_text(resp.text, url)


def crawl():
    target_netloc = urllib.parse.urlparse(TARGET_URL).netloc
    queue = deque([(TARGET_URL, 0)])

    print(f"[*] Bắt đầu quét: {TARGET_URL} (độ sâu tối đa: {MAX_DEPTH})\n")

    while queue and len(visited_pages) < MAX_PAGES:
        url, depth = queue.popleft()
        extract_links_and_apis(url, target_netloc, queue, depth)


def print_report():
    print("\n" + "=" * 60)
    print(f" TÌM THẤY {len(found_urls)} URL NỘI BỘ")
    print("=" * 60)
    for u in sorted(found_urls):
        print(u)

    print("\n" + "=" * 60)
    print(f" TÌM THẤY {len(found_apis)} API ENDPOINT (TỔNG)")
    print("=" * 60)
    for a in sorted(found_apis):
        print(a)

    print("\n" + "=" * 60)
    print(f" ⭐ {len(priority_apis)} API NGHI LIÊN QUAN MÃ LỚP / GIÁO VIÊN / LỊCH HỌC")
    print("=" * 60)
    for a in sorted(priority_apis):
        print(a)

    print("\n" + "=" * 60)
    print(f" 🔑 {len(interesting_json_keys)} JSON KEY GỢI Ý (xuất hiện trong response/JS)")
    print("=" * 60)
    for k in sorted(interesting_json_keys):
        print(k)

    print(
        "\n[!] Đây chỉ là danh sách GỢI Ý dựa trên regex — cần kiểm tra thủ công "
        "(mở từng URL / gọi thử bằng requests) để xác nhận endpoint thật, "
        "có yêu cầu xác thực hay không, và có được phép truy cập hay không."
    )


if __name__ == "__main__":
    crawl()
    print_report()