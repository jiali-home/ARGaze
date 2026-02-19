Generated files:
- basic_stats.csv
- freq_by_task.csv, freq_by_site.csv, freq_by_participant.csv
- pivot_task_vs_site.csv
- bar_by_task.png, bar_by_site.png, bar_by_participant.png
- heatmap_task_vs_site.png, heatmap_site_vs_task.png
- boxplot_by_task.png, boxplot_by_site.png, boxplot_by_participant.png

Drill-down (Site-Task) in explore/drilldown/:
- pivot_site_vs_task.csv, heatmap_site_vs_task.png
- top_site_task_pairs.csv
- <site>__<task>/participants_distribution.csv/.png

Participant-centric in explore/participant_centric/:
- selected_participants.csv (top + mid)
- <participant>/pivot_site_vs_task.csv/.png

Detected columns: task=task_name, site=university_name, participant=participant_uid
