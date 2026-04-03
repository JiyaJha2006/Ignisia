from collections import defaultdict

def group_by_cluster(data):
    clusters = defaultdict(list)

    for row in data:
        clusters[row["cluster_id"]].append(row)

    return clusters