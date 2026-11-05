"""Read-only source/license/build-lock admission; no solver/model/data execution."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import importlib.metadata as md
import io
import json
from pathlib import Path
import platform
import sys

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def encode(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def source_manifest(dist,prefix):
    entries=[]
    for name,recorded,size in csv.reader(io.StringIO(dist.read_text("RECORD"))):
        path=Path(dist.locate_file(name)).resolve()
        assert path.is_relative_to(prefix), "distribution file escapes runtime"
        relative=path.relative_to(prefix).as_posix()
        if not recorded or not relative.lower().startswith("lib/site-packages/") or relative.endswith(".pyc") or relative.endswith(("/INSTALLER","/REQUESTED","/direct_url.json","/RECORD")): continue
        entries.append([relative,recorded,int(size)])
    return sorted(entries)

def check(root,expected,full=False):
    profile_path=root/"provenance.json"
    assert digest(profile_path)==expected,"external profile identity differs; local relabel cannot authorize itself"
    raw=profile_path.read_bytes(); assert b"\r" not in raw and raw.endswith(b"\n"),"UTF8/LF profile required"
    profile=json.loads(raw);prefix=Path(sys.prefix).resolve();runtime=profile["runtime"]
    assert (platform.python_implementation(),platform.python_version(),sys.platform,platform.machine())==(runtime["implementation"],runtime["python"],runtime["platform"],runtime["machine"]),"runtime identity mismatch"
    assert "include-system-site-packages = false" in (prefix/"pyvenv.cfg").read_text().lower(),"shared site packages refused"
    assert digest(root/"requirements.lock")==profile["requirements_sha256"],"requirement bytes differ"
    assert digest(root/"defaults.json")==profile["defaults_sha256"],"default settings bytes differ"
    installed={d.metadata["Name"].lower().replace("_","-"):d for d in md.distributions()}
    expected_names={r["name"].lower().replace("_","-") for r in profile["distributions"]}
    assert set(installed)==expected_names,"missing or unexpected installed distribution"
    file_count=0
    for row in profile["distributions"]:
        dist=installed[row["name"].lower().replace("_","-")];assert dist.version==row["version"],"version mismatch"
        for key in ["metadata","wheel"]:
            target=(prefix/row[key+"_relative_path"]).resolve();assert target.is_relative_to(prefix)
            assert digest(target)==row[key+"_sha256"],key+" identity differs"
        entries=source_manifest(dist,prefix)
        assert len(entries)==row["source_file_count"] and hashlib.sha256(encode(entries)).hexdigest()==row["source_manifest_sha256"],"package source manifest differs"
        for license in row["licenses"]:
            path=(prefix/license["installed_relative_path"]).resolve();public=(root/license["profile_relative_path"]).resolve()
            assert path.is_relative_to(prefix) and public.is_relative_to(root.resolve()),"license path escape"
            assert digest(path)==digest(public)==license["sha256"],"license identity differs"
        if full:
            for relative,recorded,size in entries:
                path=(prefix/relative).resolve();assert path.is_relative_to(prefix)
                algorithm,expected_hash=recorded.split("=",1);h=hashlib.new(algorithm)
                with path.open("rb") as handle:
                    for chunk in iter(lambda:handle.read(1024*1024),b""): h.update(chunk)
                actual=base64.urlsafe_b64encode(h.digest()).decode().rstrip("=")
                assert actual==expected_hash and path.stat().st_size==size,"installed source bytes differ: "+relative
                file_count+=1
    for origin in profile["imported_origins"].values():
        path=(prefix/origin["relative_path"]).resolve();assert path.is_relative_to(prefix)
        assert digest(path)==origin["sha256"],"import origin bytes differ"
    return dict(distributions=len(installed),full_source_files_checked=file_count,full_files=full,read_only=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--check",action="store_true",required=True);parser.add_argument("--expected-profile-sha256",required=True);parser.add_argument("--profile",type=Path,default=Path(__file__).resolve().parent);parser.add_argument("--full-files",action="store_true");args=parser.parse_args()
    print(json.dumps(check(args.profile.resolve(),args.expected_profile_sha256,args.full_files),sort_keys=True))
if __name__=="__main__":main()
