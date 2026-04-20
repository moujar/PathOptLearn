import yt_dlp
from app.config import YT_MAX_RESULTS


def search_youtube(query: str, max_results: int = YT_MAX_RESULTS) -> list[dict]:
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    videos = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            for entry in info.get("entries", []):
                d      = int(entry.get("duration") or 0)
                vid_id = entry.get("id", "")
                videos.append({
                    "id":       vid_id,
                    "title":    entry.get("title", ""),
                    "channel":  entry.get("channel") or entry.get("uploader", ""),
                    "duration": f"{d // 60}:{d % 60:02d}",
                    "views":    entry.get("view_count"),
                    "desc":     (entry.get("description") or "")[:200],
                    "url":      f"https://www.youtube.com/watch?v={vid_id}",
                    "thumb":    f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg",
                })
    except Exception as e:
        print(f"[YouTube error] {e}")
    return videos
