# SPDX-License-Identifier: Apache-2.0
"""Example states and questions for the nouls demo.

Short examples show the basic comparison. Long examples (hundreds to ~1.5k
tokens of state) show off the endpoint batching: every question shares the
state as a cached prefix, so the longer the state, the larger the share of
each question's prompt that vLLM reuses instead of recomputing.

All companies, people, and events are fictional.
"""

import json

SHORT = [
    (
        "Short · double charge",
        "I was charged twice for my subscription this month and nobody has "
        "answered my last two emails. Please fix this today or cancel my account.",
        "Is this about billing?\nIs the customer angry?\n"
        "Does the customer want a refund?\nIs this a bug report?",
    ),
    (
        "Short · damaged order (JSON)",
        '{"order_id": "A-1042", "status": "delivered", "delivered_at": '
        '"2026-09-20", "customer_message": "The box arrived crushed and the '
        'mug inside is in pieces."}',
        "Was the item damaged?\nHas the order been delivered?\n"
        "Is the customer asking to change the shipping address?",
    ),
    (
        "Short · false claim",
        "The Eiffel Tower was completed in 1889 and is located in Berlin.",
        "Is this statement entirely accurate?\nDoes the statement mention a year?\n"
        "Is the Eiffel Tower in France?",
    ),
]

_SUPPORT_THREAD = """\
Ticket #88213 · Brightline Home Internet · Priority: normal · Channel: email

--- Message 1 · Customer · Sep 2, 08:14 ---
Hi, my internet has been dropping several times a day for about a week. It
goes out for 5 to 20 minutes at a time, usually in the evening. I work from
home, so this is a real problem. I've already restarted the modem and router
more times than I can count. Account holder: Dana Whitfield, service address
41 Larch Street, Unit 3. Can someone please look into this?

--- Message 2 · Agent (Priya, Tier 1) · Sep 2, 11:02 ---
Hi Dana, thanks for reaching out and sorry for the trouble. I ran a remote
line test and I can see intermittent signal loss on your connection. I've
scheduled a technician visit for Thursday, Sep 5, between 1 and 5 pm. The
technician will check the line from the street to your unit. Please make sure
someone over 18 is home. Is there anything else I can help with?

--- Message 3 · Customer · Sep 2, 11:20 ---
Thursday works. I'll be home all afternoon.

--- Message 4 · Customer · Sep 5, 17:48 ---
Nobody came. I waited the entire window and got no call, no text, nothing. I
took the afternoon off work for this. The internet dropped four more times
today while I was waiting. What happened?

--- Message 5 · Agent (Marcus, Tier 1) · Sep 6, 09:30 ---
Hi Dana, I'm very sorry. I can see the technician was reassigned to an urgent
outage and your appointment was not rebooked, which should not have happened.
I've rescheduled you for Tuesday, Sep 10, 8 am to noon, and flagged it as a
priority visit. For the missed appointment and the ongoing disruption, I've
applied a one-time credit of $40 to your account; you'll see it on your next
statement.

--- Message 6 · Customer · Sep 6, 10:02 ---
Fine. Tuesday morning. Please make sure someone actually shows up this time.

--- Message 7 · Customer · Sep 10, 13:15 ---
The technician came this morning, replaced the splitter outside, and said the
line looked good afterwards. I appreciated that he explained what he was
doing. Unfortunately the connection dropped again at 12:40, about an hour
after he left, for roughly ten minutes. So it's not fixed.

--- Message 8 · Agent (Priya, Tier 1) · Sep 10, 15:47 ---
Thanks for letting us know, Dana. Since the drops continued after the
splitter replacement, I've opened a network ticket so our engineering team can
check the node that serves your street. These investigations usually take
three to five business days. I'll update you as soon as I hear back.

--- Message 9 · Customer · Sep 16, 19:05 ---
It's been almost a week and nothing has changed. The internet went out three
times during a client call today and I had to finish it on my phone. I've now
been dealing with this for more than two weeks. I'm also still waiting to see
the $40 credit: my new statement arrived today and it isn't on there.

If this isn't fixed by the end of the week, I'm cancelling my service and
switching providers, and I'll be filing a complaint with the state utilities
commission. I'd also like to know whether I'll be charged the early
termination fee if I leave because you can't provide working service.

--- Message 10 · Agent (Marcus, Tier 1) · Sep 17, 08:55 ---
Hi Dana, I completely understand your frustration. I checked the network
ticket and engineering has confirmed a failing amplifier at the node serving
Larch Street; a repair is scheduled for Sep 19. I'm looking into the missing
credit now. Regarding the early termination fee, I'm not able to make that
decision at my level, so I'll need to check with my team.
"""

_SUPPORT_QUESTIONS = (
    "Has the customer been promised a bill credit?\n"
    "Did the first scheduled technician visit take place?\n"
    "Is the customer threatening to cancel their service?\n"
    "Has the connection problem been resolved?\n"
    "Does the customer mention filing a complaint with a regulator?\n"
    "Has the root cause of the outages been identified?\n"
    "Is the customer asking about an early termination fee?\n"
    "Did the customer say anything positive about the technician?"
)

_ORDER_RECORD = {
    "order_id": "WX-55190-K",
    "placed_at": "2026-09-03T14:22:05Z",
    "channel": "mobile_app",
    "customer": {
        "customer_id": "C-204417",
        "name": "Rafael Okonkwo",
        "email_verified": True,
        "member_since": "2021-06-11",
        "lifetime_orders": 23,
        "loyalty_tier": "gold",
        "notes": "Prefers text updates. Previously reported one missing item (2024).",
    },
    "items": [
        {"sku": "KT-8812", "name": "Cast iron dutch oven, 5.5 qt", "qty": 1, "unit_price": 89.00},
        {"sku": "KT-1045", "name": "Silicone trivet set (3)", "qty": 1, "unit_price": 14.50},
        {"sku": "KT-3307", "name": "Enameled saucepan, 2 qt", "qty": 2, "unit_price": 42.00},
    ],
    "payment": {
        "method": "credit_card",
        "subtotal": 187.50,
        "shipping": 0.00,
        "tax": 15.47,
        "total": 202.97,
        "captured": True,
        "refunds": [],
    },
    "shipping": {
        "address_at_checkout": "1180 Birchwood Ave, Apt 2B, Milltown, OR 97402",
        "current_address": "77 Harbor View Rd, Milltown, OR 97405",
        "method": "standard_ground",
        "promised_delivery": "2026-09-09",
        "carrier": "Parcelway",
        "tracking": "PW7730018842216",
    },
    "events": [
        {"at": "2026-09-03T14:22:05Z", "type": "order_placed"},
        {"at": "2026-09-03T14:22:09Z", "type": "payment_captured", "amount": 202.97},
        {"at": "2026-09-03T18:40:31Z", "type": "customer_message", "text": "Hi, I'm moving this weekend. Can you send it to 77 Harbor View Rd instead?"},
        {"at": "2026-09-03T19:05:12Z", "type": "address_changed", "by": "agent", "to": "77 Harbor View Rd, Milltown, OR 97405"},
        {"at": "2026-09-04T09:15:44Z", "type": "picked", "warehouse": "PDX-2"},
        {"at": "2026-09-04T13:02:10Z", "type": "packed", "packages": 1, "weight_lb": 19.2},
        {"at": "2026-09-04T16:48:00Z", "type": "shipped", "carrier": "Parcelway"},
        {"at": "2026-09-05T02:11:37Z", "type": "carrier_scan", "location": "Portland, OR hub"},
        {"at": "2026-09-06T21:30:05Z", "type": "carrier_scan", "location": "Eugene, OR facility"},
        {"at": "2026-09-08T07:45:12Z", "type": "carrier_exception", "reason": "Weather delay; regional flooding"},
        {"at": "2026-09-09T10:00:00Z", "type": "promised_date_missed"},
        {"at": "2026-09-09T17:22:48Z", "type": "customer_message", "text": "It was supposed to arrive today. Any update?"},
        {"at": "2026-09-09T18:01:30Z", "type": "agent_reply", "text": "Sorry for the delay. The carrier reports weather disruption in your area; we expect movement within 2-3 days."},
        {"at": "2026-09-11T08:14:55Z", "type": "carrier_scan", "location": "Eugene, OR facility"},
        {"at": "2026-09-12T06:30:20Z", "type": "out_for_delivery"},
        {"at": "2026-09-12T15:47:03Z", "type": "delivered", "location": "front porch", "photo": True},
        {"at": "2026-09-12T19:10:41Z", "type": "customer_message", "text": "Got it, thanks. The dutch oven lid has a big chip in the enamel though."},
        {"at": "2026-09-12T19:12:00Z", "type": "customer_upload", "files": ["lid_chip_1.jpg", "lid_chip_2.jpg"]},
        {"at": "2026-09-13T10:25:36Z", "type": "return_requested", "sku": "KT-8812", "reason": "damaged", "resolution_requested": "replacement"},
        {"at": "2026-09-13T10:26:02Z", "type": "return_label_issued", "carrier": "Parcelway"},
        {"at": "2026-09-14T09:40:00Z", "type": "agent_note", "text": "Replacement reserved from PDX-2; ships when return scan is received. No refund needed if replacement accepted."},
    ],
    "return_status": "awaiting_customer_dropoff",
}

_INCIDENT_REPORT = """\
INCIDENT REPORT · INC-4471 · Ledgerly Payments API · Severity: SEV-2
Status: Monitoring · Incident commander: J. Castellanos · Report owner: A. Brandt

SUMMARY
Between 09:12 and 10:47 UTC on Sep 18, a share of card authorization requests
to the Payments API failed with HTTP 503 errors. At peak, 31% of requests in
the us-west region failed; other regions were unaffected. Merchants saw
declined checkouts they had to retry. No payment was charged twice, and no
transaction or customer data was lost or corrupted; failed requests were
rejected before any money moved.

TIMELINE (UTC)
09:05  Deploy 2026.09.18-3 rolls out to us-west. It includes a change to the
       connection-pool configuration for the card-network gateway
       (max_connections lowered from 512 to 64 as part of a cleanup).
09:12  Error rate in us-west begins climbing. Automated alert fires at 09:14.
09:16  On-call engineer acknowledges. Initial suspicion: upstream card network.
09:31  Card network status page reports no issues. Investigation moves to
       our gateway service.
09:44  Engineers notice gateway pods saturating their connection pools;
       requests queue and time out after 10 s, surfacing as 503s.
09:58  Status page updated: "Elevated errors for card authorizations in
       us-west. Investigating." Email notice sent to affected merchants.
10:20  Config diff for deploy 2026.09.18-3 identified as the likely cause.
10:29  Decision to roll back rather than hotfix the config.
10:38  Rollback to 2026.09.18-2 completes in us-west.
10:47  Error rate back to baseline (<0.1%). Incident moved to monitoring.
11:30  Status page updated: "Resolved; monitoring." Follow-up email to
       affected merchants with a summary.

SAMPLE LOG LINES (gateway, us-west)
09:13:02 WARN  pool=cardnet active=64/64 waiting=212 wait_ms=8431
09:13:05 ERROR request_id=7f3a… upstream_timeout after 10000ms -> 503
09:40:47 WARN  pool=cardnet active=64/64 waiting=1904 wait_ms=9988
10:39:15 INFO  config reloaded: cardnet.max_connections=512
10:41:02 INFO  pool=cardnet active=138/512 waiting=0 wait_ms=2

ROOT CAUSE
The connection-pool limit for the card-network gateway was lowered from 512
to 64 in a configuration cleanup. The reviewer approved it believing the old
value was unused. Under normal morning traffic in us-west, 64 connections were
far too few, so requests queued until they timed out.

IMPACT
- Duration: 95 minutes (09:12-10:47 UTC).
- About 48,200 authorization requests failed in us-west; most were retried
  successfully by merchant integrations.
- 2 enterprise merchants opened support tickets; both have been contacted.
- No data loss. No duplicate charges. No security impact.

ACTION ITEMS
1. Add a staging load test that exercises the gateway at production
   concurrency before config changes ship. (Owner: platform, due Oct 2)
2. Alert on connection-pool saturation, not only on error rate.
   (Owner: SRE, due Sep 25)
3. Require a second approver for connection and timeout settings.
   (Owner: A. Brandt, due Sep 30)
4. Publish an external post-incident summary. (Owner: comms, not started)
"""

_INCIDENT_QUESTIONS = (
    "Is the incident fully closed?\n"
    "Was any customer data lost?\n"
    "Was the root cause a configuration change?\n"
    "Did the outage last longer than one hour?\n"
    "Was a rollback performed?\n"
    "Were affected merchants notified?\n"
    "Were all regions affected?\n"
    "Has the external post-incident summary been published?"
)

LONG = [
    ("Long · ISP support thread", _SUPPORT_THREAD, _SUPPORT_QUESTIONS),
    (
        "Long · order record + event log (JSON)",
        json.dumps(_ORDER_RECORD, indent=2),
        "Was the package delivered?\n"
        "Was the package delivered after the promised date?\n"
        "Has a refund been issued?\n"
        "Is this a repeat customer?\n"
        "Did the customer change the shipping address after ordering?\n"
        "Is there an open return request?\n"
        "Did the customer ask for a replacement rather than a refund?\n"
        "Was the whole order damaged?",
    ),
    ("Long · incident report", _INCIDENT_REPORT, _INCIDENT_QUESTIONS),
]

# 50 yes/no questions on the order record, each answerable from it, with the
# expected answer alongside for reference (33 yes, 17 no). Stress-tests the
# batching: one primed question, then 49 in parallel on the cached state.
ORDER_50_QUESTIONS = [
    ('Was the order placed through the mobile app?', True),
    ('Was the order placed on the website?', False),
    ("Is the customer's email verified?", True),
    ('Has the customer been a member since before 2022?', True),
    ("Is this the customer's first order?", False),
    ('Is the customer in the gold loyalty tier?', True),
    ('Does the customer prefer email updates over text?', False),
    ('Has the customer reported a missing item before?', True),
    ('Did the order include a cast iron dutch oven?', True),
    ('Did the order include a frying pan?', False),
    ('Were two enameled saucepans ordered?', True),
    ('Does the order have three line items?', True),
    ('Was the dutch oven the most expensive item?', True),
    ('Is the order total over $200?', True),
    ('Was shipping free?', True),
    ('Was tax charged on the order?', True),
    ('Was the payment made with PayPal?', False),
    ('Has the payment been captured?', True),
    ('Has any refund been issued?', False),
    ('Did the customer ask to change the shipping address?', True),
    ('Was the address changed by an agent?', True),
    ('Was the address changed before the order shipped?', True),
    ('Is the current shipping address on Birchwood Ave?', False),
    ('Was the order shipped with express shipping?', False),
    ('Was Parcelway the carrier?', True),
    ('Was the order picked from the PDX-2 warehouse?', True),
    ('Did the order ship in more than one package?', False),
    ('Did the package weigh more than 15 lb?', True),
    ('Did the package ship on the day it was ordered?', False),
    ('Was the package scanned at a Portland hub?', True),
    ('Did weather cause a shipping delay?', True),
    ('Was the delay caused by a snowstorm?', False),
    ('Was the promised delivery date September 9?', True),
    ('Was the promised delivery date met?', False),
    ('Did the customer contact support about the delay?', True),
    ("Did support reply to the customer's delay message?", True),
    ('Was the package delivered?', True),
    ('Was the package delivered to a front porch?', True),
    ('Is there a photo of the delivery?', True),
    ('Was the package delivered on September 12?', True),
    ('Was the package lost in transit?', False),
    ('Did the customer report damage?', True),
    ('Was the saucepan damaged?', False),
    ('Did the customer upload photos of the damage?', True),
    ('Did the customer request a return?', True),
    ('Did the customer ask for a refund rather than a replacement?', False),
    ('Has a return label been issued?', True),
    ('Has the customer dropped off the return yet?', False),
    ('Has a replacement been reserved?', True),
    ('Has the replacement already shipped?', False),
]

STRESS = [
    (
        "Stress · 50 questions on the order record (JSON)",
        json.dumps(_ORDER_RECORD, indent=2),
        "\n".join(q for q, _ in ORDER_50_QUESTIONS),
    ),
]

EXAMPLES = SHORT + LONG + STRESS
EXAMPLE_LABELS = [label for label, _, _ in EXAMPLES]
EXAMPLE_INPUTS = [[state, questions] for _, state, questions in EXAMPLES]
