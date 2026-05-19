TOOLS = [
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
            "description": "환불정책, 교환, 배송 등 사내정책 관련 질문에 답변합니다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
