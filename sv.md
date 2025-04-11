total_revenue_from_operations should  be a result of all extracted segment_revenues
example
"segment_revenues": {
          "Retail": 738.2,
          "Channel & Enterprise": 698.96
        },
        "total_revenue_from_operations": 1437.16,
738.2 + 698.96 should account to 1437.16

total_segment_profit_before_interest_tax should be a result of all extracted segment_results
example
"segment_results": {
          "Retail": 41.13,
          "Channel & Enterprise": 7.24
        },
        "total_segment_profit_before_interest_tax": 48.37,
41.13 + 7.24 should account to 48.37

total_assets should be sum of a result of all extracted segment_assets and unallocated_assets
"segment_assets": {
          "Retail": 616.2,
          "Channel & Enterprise": 682.59
        },
        "unallocated_assets": 392.97,
        "total_assets": 1691.76,
616.2 + 682.59 + 392.97 should account to 1691.76


total_liabilities should be sum of a result of all extracted segment_liabilities and unallocated_liabilities
"segment_liabilities": {
          "Retail": 107.54,
          "Channel & Enterprise": 602.87
        },
        "unallocated_liabilities": 141.95,
        "total_liabilities": 852.36
107.54 + 602.87 + 141.95 should account to 852.36