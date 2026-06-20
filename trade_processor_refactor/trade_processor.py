from __future__ import annotations

import json
import os
import smtplib
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Dict

import redis
import requests
from openai import OpenAI

# Allows running this file from the folder where agent.py and trade_processor.py are located.
sys.path.append(str(Path(__file__).resolve().parent))

from agent import Agent


class TradeProcessor(Agent):
    """
    Redis-based trade processing agent.

    Flow:
    1. Listen to Redis queue `trades`
    2. Parse incoming JSON
    3. Validate trade with OpenAI LLM
    4. If valid, write to `trades_outgoing`
    5. If invalid, send rejection email and write to `trades_rejected`
    """

    GRAPH = [
        {"name": "listen_for_trade", "tool": "redis_listener"},
        {"name": "parse_trade", "tool": "parse_trade"},
        {"name": "validate_trade", "tool": "llm_trade_validator"},
        {"name": "route_trade", "tool": "route_trade"},
    ]

    def __init__(self) -> None:
        self.incoming_queue = os.getenv("INCOMING_TRADE_QUEUE", "trades")
        self.outgoing_queue = os.getenv("OUTGOING_TRADE_QUEUE", "trades_outgoing")
        self.rejected_queue = os.getenv("REJECTED_TRADE_QUEUE", "trades_rejected")

        self.default_rejection_email = os.getenv("DEFAULT_REJECTION_EMAIL", "lakshmisnaird@gmail.com")

        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            db=int(os.getenv("REDIS_DB", "0")),
            decode_responses=True,
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Set it in the terminal before running trade_processor.py.")

        self.openai_client = OpenAI(api_key=api_key)
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # -----------------------------
    # Graph tools
    # -----------------------------

    def redis_listener(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Redis Listener] Waiting on queue: {self.incoming_queue}")
        _, raw_message = self.redis_client.blpop(self.incoming_queue, timeout=0)
        state["raw_message"] = raw_message
        print("[Redis Listener] Trade event received")
        return state

    def parse_trade(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            state["trade"] = json.loads(state["raw_message"])
            print(f"[Parser] Trade parsed: {state['trade'].get('trade_id', 'UNKNOWN')}")
        except json.JSONDecodeError as exc:
            state["trade"] = {"raw_message": state.get("raw_message")}
            state["is_valid"] = False
            state["validation_reason"] = f"Invalid JSON: {exc}"
            state["skip_llm_validation"] = True
            print(f"[Parser] Invalid JSON: {exc}")
        return state

    def llm_trade_validator(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("skip_llm_validation"):
            return state

        trade = state["trade"]

        prompt = f"""
You are a strict financial trade validation agent.

Validate the trade using these rules:
1. trade_id must exist and must not be empty.
2. symbol must exist and must not be empty.
3. quantity must be a positive number greater than 0.
4. price must be a positive number greater than 0.
5. side must be either BUY or SELL.
6. submitted_by_email or trader_email is optional and should not affect validity.
7. Extra fields are allowed.

Return ONLY valid JSON in this exact format:
{{
  "is_valid": true,
  "reason": "short validation result"
}}

Trade:
{json.dumps(trade, indent=2)}
""".strip()

        response = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "You validate trades and return only valid JSON. No markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        content = response.choices[0].message.content.strip()
        result = json.loads(content)

        state["is_valid"] = bool(result["is_valid"])
        state["validation_reason"] = str(result["reason"])

        print(f"[LLM Validator] valid={state['is_valid']} reason={state['validation_reason']}")
        return state

    def route_trade(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if state.get("is_valid"):
            self.redis_writer(
                self.outgoing_queue,
                {
                    "status": "VALIDATED",
                    "trade": state["trade"],
                    "validated_at": int(time.time()),
                    "validated_by": self.openai_model,
                    "validation_reason": state["validation_reason"],
                },
            )
            print(f"[Router] Valid trade sent to queue: {self.outgoing_queue}")
        else:
            rejected_payload = {
                "status": "REJECTED",
                "trade": state["trade"],
                "rejected_at": int(time.time()),
                "validated_by": self.openai_model,
                "validation_reason": state["validation_reason"],
            }

            self.redis_writer(self.rejected_queue, rejected_payload)
            self.email_sender(
                to_email=self.default_rejection_email,
                subject=f"Invalid Trade Rejected - {state['trade'].get('trade_id', 'UNKNOWN')}",
                body=self._rejection_email_body(state),
            )

            print(f"[Router] Invalid trade sent to queue: {self.rejected_queue}")
            print(f"[Router] Rejection email sent/printed for: {self.default_rejection_email}")

        state["stop"] = True
        return state

    # -----------------------------
    # Separate tools available to the processor
    # -----------------------------

    def redis_writer(self, queue_name: str, payload: Dict[str, Any]) -> None:
        self.redis_client.rpush(queue_name, json.dumps(payload))

    def email_sender(self, to_email: str, subject: str, body: str) -> None:
        smtp_server = os.getenv("SMTP_SERVER", "localhost")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")

        if not smtp_user or not smtp_password:
            print("\n[TEST MODE EMAIL]")
            print(f"To: {to_email}")
            print(f"Subject: {subject}")
            print(body)
            return

        message = EmailMessage()
        message["From"] = smtp_user
        message["To"] = to_email
        message["Subject"] = subject
        message.set_content(body)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(message)

    def web_search(self, query: str, num_results: int = 5) -> str:
        """
        Separate web-search tool. It is not used in the current static graph,
        but remains available if the graph is extended later.
        """
        serpapi_key = os.getenv("SERPAPI_KEY")
        if not serpapi_key:
            return "SERPAPI_KEY is not set."

        response = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google", "q": query, "api_key": serpapi_key, "num": num_results},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("organic_results", [])[:num_results]
        return "\n\n".join(
            f"{i}. {item.get('title', '')}\n{item.get('snippet', '')}\n{item.get('link', '')}"
            for i, item in enumerate(results, start=1)
        )

    def _rejection_email_body(self, state: Dict[str, Any]) -> str:
        return f"""Hello,

A trade was rejected by the Trade Processor Agent.

Reason:
{state["validation_reason"]}

Trade Details:
{json.dumps(state["trade"], indent=2)}

Please correct the trade and submit it again.

Regards,
Trade Processor Agent
"""


def main() -> None:
    processor = TradeProcessor()

    print("[Trade Processor Started]")
    print(f"Incoming queue: {processor.incoming_queue}")
    print(f"Outgoing queue: {processor.outgoing_queue}")
    print(f"Rejected queue: {processor.rejected_queue}")
    print(f"Rejection email: {processor.default_rejection_email}")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            processor.invoke(processor.GRAPH, {})
        except KeyboardInterrupt:
            print("\n[Trade Processor Stopped]")
            break
        except Exception as exc:
            print(f"[Error] {exc}")


if __name__ == "__main__":
    main()
