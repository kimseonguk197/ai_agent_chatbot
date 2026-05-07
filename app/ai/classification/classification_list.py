TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "상품 목록을 조회합니다. '사과 조회해줘', '상품 목록 보여줘' 등의 요청에 사용합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_orders",
            "description": "로그인한 사용자 본인의 주문 내역을 조회합니다. '내 주문', '주문 내역', '내가 주문한 것' 등의 요청에 사용합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_my_profile",
            "description": "사용자 본인의 회원정보를 조회합니다. '내 정보', '마이페이지', '내 계정' 등의 요청에 사용합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "환불정책, FAQ 등 정책 관련 질문에 답변합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_document",
            "description": "문서 작성, 메신저 전송 등 관리자/직원의 요청을 처리합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
