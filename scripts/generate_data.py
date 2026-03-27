#!/usr/bin/env python3
"""
Generate synthetic data for FreightIQ:
1. Shipments dataset (training data for XGBoost risk model)
2. Historical disruption reports (for RAG context)
"""

import os
import random
import csv
import json
from datetime import datetime, timedelta
from faker import Faker
import argparse

# Initialize Faker
fake = Faker()
Faker.seed(42)
random.seed(42)

REGIONS = ["Europe", "North America", "Asia", "South America", "Middle East", "Africa"]
CARRIERS = ["Maersk", "MSC", "CMA CGM", "Hapag-Lloyd", "Evergreen", "ONE", "FedEx", "DHL", "UPS"]
ROUTES = [
    "Shanghai → Rotterdam",
    "Shenzhen → Los Angeles",
    "Busan → Hamburg",
    "New York → London",
    "Singapore → Dubai",
    "Mumbai → Antwerp",
    "Santos → Bremerhaven"
]

EVENT_TYPES = ["labour_dispute", "weather_event", "port_congestion", "geopolitical_issue", "customs_delay"]

def generate_shipments(n: int, output_path: str):
    """Generate synthetic shipment records with risk features and target variable."""
    shipments = []
    
    for _ in range(n):
        # Base features
        route = random.choice(ROUTES)
        region = random.choice(REGIONS)
        carrier = random.choice(CARRIERS)
        
        # Risk features
        carrier_reliability = round(random.uniform(0.6, 0.99), 2)
        region_disruption_count = random.randint(0, 5)
        days_to_delivery = random.randint(1, 45)
        weather_severity = round(random.uniform(0.0, 1.0), 2)
        route_risk_score = round(random.uniform(0.1, 0.9), 2)
        cargo_value_usd = round(random.uniform(10_000, 500_000), 2)
        news_sentiment_score = round(random.uniform(-1.0, 1.0), 2)
        
        # Generate the target variable realistically based on the features
        # Higher risk if reliability is low, weather is severe, high disruptions, negative sentiment
        risk_likelihood = (
            (1.0 - carrier_reliability) * 0.2 +
            (weather_severity) * 0.25 +
            (region_disruption_count / 5.0) * 0.2 +
            (route_risk_score) * 0.15 +
            ((1.0 - news_sentiment_score) / 2.0) * 0.2
        )
        
        # Add some random noise
        risk_likelihood = min(max(risk_likelihood + random.uniform(-0.1, 0.1), 0.0), 1.0)
        
        # Binary target: 1 = Delayed, 0 = On Time
        is_delayed = 1 if risk_likelihood > 0.55 else 0
        
        # Also compute a synthetic "delay_days" if delayed
        delay_days = random.randint(1, 14) if is_delayed else 0

        shipments.append({
            "shipment_id": f"SHP-{fake.unique.random_int(min=10000, max=99999)}",
            "route": route,
            "region": region,
            "carrier": carrier,
            "carrier_reliability": carrier_reliability,
            "region_disruption_count": region_disruption_count,
            "days_to_delivery": days_to_delivery,
            "weather_severity": weather_severity,
            "route_risk_score": route_risk_score,
            "cargo_value_usd": cargo_value_usd,
            "news_sentiment_score": news_sentiment_score,
            "is_delayed": is_delayed,
            "delay_days": delay_days
        })
    
    # Write to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=shipments[0].keys())
        writer.writeheader()
        writer.writerows(shipments)
    print(f"✅ Generated {n} synthetic shipments at {output_path}")


def generate_disruptions(n: int, output_path: str):
    """Generate synthetic historical disruption reports for RAG."""
    disruptions = []
    
    for _ in range(n):
        event_type = random.choice(EVENT_TYPES)
        region = random.choice(REGIONS)
        port = f"Port of {fake.city()}"
        
        # Generate text description suited for RAG
        if event_type == "labour_dispute":
            txt = f"Dockworkers at {port} in {region} have declared an indefinite strike demanding better pay. Operations have halted completely, leading to a massive buildup of vessels at anchorage. Carriers like {random.choice(CARRIERS)} are rerouting via neighboring ports."
        elif event_type == "weather_event":
            txt = f"A category 4 typhoon has forced the closure of {port}. Severe flooding in {region} has washed out rail links to the hinterland. Expect delays of 7-14 days for all active {random.choice(ROUTES)} shipments."
        elif event_type == "port_congestion":
            txt = f"Volume surges ahead of the holiday season have crippled operations at {port}. Wait times at the {region} terminal are exceeding 12 days. Container yard utilization is at 98%."
        elif event_type == "geopolitical_issue":
            txt = f"Naval blockades in {region} have forced vessels to abandon the {random.choice(ROUTES)} route. Ships are diverting around the cape, adding 10-15 days to transit times and $2000 per TEU in surcharges."
        else:
            txt = f"A cyberattack on the customs clearing system in {region} has paralyzed {port}. Thousands of containers are sitting without clearance. Standard mitigation strategy: discharge cargo at alternate ports and truck across border."
            
        disruptions.append({
            "id": fake.uuid4(),
            "date": fake.date_between(start_date='-2y', end_date='today').isoformat(),
            "region": region,
            "event_type": event_type,
            "port": port,
            "description": txt,
            "source": random.choice(["Lloyds List", "JOC", "SupplyChainDive", "Internal Report"])
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(disruptions, f, indent=2)
    print(f"✅ Generated {n} synthetic disruption reports at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data for FreightIQ")
    parser.add_argument("--shipments", type=int, default=1000, help="Number of shipments to generate (default: 1000)")
    parser.add_argument("--disruptions", type=int, default=50, help="Number of disruptions to generate (default: 50)")
    parser.add_argument("--output", type=str, default="data/synthetic/", help="Output directory")
    args = parser.parse_args()

    # Generate slightly larger training dataset by default (1000 vs 100) for better XGBoost training
    generate_shipments(args.shipments, os.path.join(args.output, "shipments.csv"))
    generate_disruptions(args.disruptions, os.path.join(args.output, "disruptions.json"))


if __name__ == "__main__":
    main()
