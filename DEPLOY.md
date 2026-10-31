# Deploying FitNotes Coach as a free public demo

This guide puts the app on the internet at an `https://` address, for free, on an
**Oracle Cloud "Always Free" Ampere (ARM) virtual machine**, using **Docker Compose**.
No previous server experience is assumed. Budget about an hour the first time,
most of it waiting.

What you end up with:

```
visitor ──HTTPS──▶ Caddy (ports 80/443, automatic certificate)
                      │
                      ▼  internal Docker network only
                   app container
                   ├─ UI + same-origin proxy  :3000
                   ├─ API                     127.0.0.1:8000 (never exposed)
                   └─ MCP tool server         (child process)
                   data volume: workout DB, RAG index, memory, settings
```

The demo runs with `DEMO_MODE=1`:

* It uses a **synthetic** workout history (`scripts/make_demo_db.py`), about 12
  weeks of push/pull/legs training by a person who doesn't exist. No real
  personal data is involved.
* There is **one shared session**. Every visitor talks to the same coach and
  sees the same chat history.
* Uploading a database or articles, clearing logs, changing settings and
  reloading the DB are **turned off** (the server answers 403 and the UI says why).
* Each visitor IP gets a small **question quota** (3 per minute, 30 per day by
  default), so one person can't use up your Gemini quota.

---

## 0. What you need

* A credit or debit card for Oracle's identity check. Always Free resources
  aren't charged.
* A **Google account** for a free Gemini API key.
* About 30 minutes for Oracle's sign-up checks.
* Optional: a domain name. Without one, use the free `sslip.io` hostname
  explained in step 6.

---

## 1. Create an Oracle Cloud account

1. Go to <https://www.oracle.com/cloud/free/> and click **Start for free**.
2. Fill in the form. Choose your **Home Region** carefully: it **cannot be
   changed later**, and Always Free compute only runs in your home region.
   Big regions (Ashburn, Phoenix, Frankfurt, London) often have no free ARM
   capacity, so a quieter region near you usually works better.
3. Verify your card. Oracle may show a small temporary authorization hold.
   Always Free resources stay free.
4. Wait for the "Your account is ready" email, then sign in at
   <https://cloud.oracle.com>.

> **Free trial vs Always Free.** New accounts also get 30 days of trial credit.
> When the trial ends, anything above the Always Free limits is stopped and
> later deleted. If you stay **within** the limits below from day one, nothing
> changes when the trial ends.

---

## 2. Know the Always Free limits (changed in 2026)

In 2026 Oracle **halved** the Always Free Ampere A1 allowance:

| | Before 2026 | Now |
|---|---|---|
| Ampere A1 OCPUs (total, all instances) | 4 | **2** |
| Ampere A1 memory (total) | 24 GB | **12 GB** |
| Monthly OCPU-hours | 3,000 | 1,500 |

Reports differ on the exact date (mid-June or mid-August 2026). Check the
current numbers on Oracle's [Always Free resources page](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topics-Always_Free_Resources.htm)
before you create anything. Accounts that were above the new limit have had
instances disabled.

This app needs **one VM with 2 OCPU and 12 GB**, which is the whole allowance.
It runs comfortably on that: the app container idles at roughly 750 MB of
RAM, and Caddy at about 20 MB.

Oracle may also **reclaim idle Always Free VMs** (very low CPU, network and
memory use over 7 days). A demo that gets the occasional visitor usually
counts as active, but if yours sits unused for weeks, check that it's still
there.

---

## 3. Create an SSH key

You use an SSH key to log in to the VM. Run this on **your own computer**:
macOS/Linux Terminal, or Windows PowerShell, where `ssh-keygen` is built in.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/oracle_fitnotes -C "fitnotes-demo"
```

Press Enter twice to skip the passphrase, or set one. This creates two files:

* `~/.ssh/oracle_fitnotes` is your **private** key. Never share it.
* `~/.ssh/oracle_fitnotes.pub` is the **public** key. You paste this into Oracle.

Print the public key so you can copy it:

```bash
cat ~/.ssh/oracle_fitnotes.pub
```

---

## 4. Create the VM (Ampere A1 Flex)

In the Oracle console: **☰ menu → Compute → Instances → Create instance**.

1. **Name:** `fitnotes-demo`.
2. **Placement:** leave the default availability domain for now (see
   "Out of capacity" below).
3. **Image and shape → Edit**:
   * **Change image** → **Canonical Ubuntu** → **24.04**. Make sure it's the
     standard build, not "Minimal", and its architecture is **aarch64**.
   * **Change shape** → **Virtual machine** → **Ampere** →
     **VM.Standard.A1.Flex**. Set **OCPUs = 2** and **Memory = 12 GB**.
     The page should show the "Always Free-eligible" label.
4. **Networking:** select **Create new virtual cloud network** and **Create new
   public subnet**, and make sure **Assign a public IPv4 address** is checked.
5. **Add SSH keys:** **Paste public keys** → paste the contents of
   `oracle_fitnotes.pub`.
6. **Boot volume:** the default (~47 GB) is fine. Always Free includes 200 GB
   of block storage in total.
7. Click **Create**. After a minute or two the state turns **Running**. Copy
   the **Public IP address** from the instance page; this guide calls it
   `<IP>` from here on.

### "Out of capacity for shape VM.Standard.A1.Flex"

This is common, and it means the data center has no free ARM capacity at the
moment. It isn't a problem with your account. Things that help, roughly in
this order:

1. **Try another availability domain** (AD-1/2/3, in Placement) if your region
   has more than one.
2. **Try again later.** Capacity frees up during the day, and early morning
   or late night in the region's time zone tends to work best.
3. **Start smaller** (1 OCPU / 6 GB) and **resize** to 2 / 12 later
   (Instance → **Edit** → shape), once capacity appears.
4. **Upgrade the account to Pay As You Go.** Always Free resources stay free
   and PAYG accounts get priority for capacity. **Only do this if you
   understand that anything beyond the free limits will then be billed.**
   Set a budget alert (☰ → Billing → Budgets) if you do.

---

## 5. Log in to the VM

```bash
ssh -i ~/.ssh/oracle_fitnotes ubuntu@<IP>
```

Type `yes` the first time to trust the host. You're now on the VM, and every
following command runs there unless stated otherwise.

Update the system first:

```bash
sudo apt-get update && sudo apt-get -y upgrade
```

If it says a reboot is required, run `sudo reboot`, wait a minute, and SSH in
again.

---

## 6. Open ports 80 and 443 (two firewalls)

Oracle blocks web traffic in **two separate places**, and you must open
**both**. Opening just one of them is the most common reason a new Oracle
deployment "doesn't load".

### 6a. Cloud firewall: the VCN security list (in the web console)

**☰ → Networking → Virtual cloud networks →** your VCN **→ Subnets →** the
public subnet **→ Security Lists → Default Security List → Add Ingress Rules**.
Some console versions put security lists under a **Security** tab on the VCN
page instead.

Add these rules. In each one, **Source Type** = CIDR, **Source CIDR** =
`0.0.0.0/0`, and **Stateless** stays unchecked:

| IP protocol | Destination port range | Description |
|---|---|---|
| TCP | `80`  | HTTP (needed for the certificate) |
| TCP | `443` | HTTPS |
| UDP | `443` | HTTP/3 (optional) |

Port 22 (SSH) is already open by default. Leave it as it is.

### 6b. VM firewall: iptables (on the VM)

Oracle's Ubuntu images ship with an iptables policy that **rejects everything
except SSH**, so the VCN rule alone isn't enough.

**Do this before installing Docker in step 7.** That way the rules you save
don't capture Docker's own rules.

Look at the current INPUT chain:

```bash
sudo iptables -L INPUT -n --line-numbers
```

You'll see an `ACCEPT ... tcp dpt:22` line, followed by a
`REJECT ... reject-with icmp-host-prohibited` line, usually at position 5 or 6.
The new rules must go **above** that REJECT line. Insert them at the REJECT
line's number. The commands below use 6, so replace it if yours differs:

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
```

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
```

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p udp --dport 443 -j ACCEPT
```

Check the result. The three new ACCEPT lines must appear **before** the REJECT
line:

```bash
sudo iptables -L INPUT -n --line-numbers
```

Make the rules survive reboots (the `netfilter-persistent` package is
pre-installed on Oracle's Ubuntu images):

```bash
sudo netfilter-persistent save
```

The rules are now saved in `/etc/iptables/rules.v4`. If you ever need to
re-add them after Docker is installed, edit that file by hand instead: add
the three `-A INPUT ... --dport 80/443 -j ACCEPT` lines just above
`-A INPUT -j REJECT --reject-with icmp-host-prohibited`, then reboot. Don't run
`netfilter-persistent save` while Docker is running.

> Don't disable the firewall wholesale (`iptables -F` or `ufw disable`
> recipes). Oracle's rules also protect the instance metadata service.

---

## 7. Install Docker

Use Docker's official install script, which supports Ubuntu on ARM:

```bash
curl -fsSL https://get.docker.com | sudo sh
```

Let the `ubuntu` user run Docker without `sudo`:

```bash
sudo usermod -aG docker ubuntu
```

Log out (`exit`) and SSH in again so the group change applies, then check:

```bash
docker run --rm hello-world
```

```bash
docker compose version
```

---

## 8. Get the code

```bash
git clone https://github.com/SwapnilAKale/FitNotes-Agentic-AI-Coach.git
```

```bash
cd FitNotes-Agentic-AI-Coach
```

```bash
git checkout Multi-Agent-System
```

(Use whichever branch holds the Docker files, if it's been merged elsewhere.)

---

## 9. Get a Gemini key that can't run up a bill

1. Go to <https://aistudio.google.com> → **Get API key** → **Create API key**.
2. Create it in a **new Google Cloud project with no billing account
   attached**. A key on an unbilled project uses the **free tier only**: when
   the free quota runs out, requests fail with "429 quota exceeded" (the UI
   shows a friendly message) instead of being charged.
3. **Never enable billing** on that project. If AI Studio shows "Paid tier"
   next to the key, it's attached to billing, so create a new project for the
   demo instead.

The app's models (`gemini-3.1-flash-lite`) are available on the free tier.
Free-tier requests may be used by Google to improve its products, which is
fine for synthetic demo data. Don't use that key for your own real data.

---

## 10. Configure `.env`

```bash
cp .env.example .env
```

```bash
nano .env
```

Set at least these two values (save with Ctrl+O, Enter, then exit with Ctrl+X):

```dotenv
GEMINI_API_KEY=AIza...your key...
DOMAIN=203-0-113-7.sslip.io
```

**`DOMAIN`**

* **No domain?** Use your VM's public IP with **dashes** plus `.sslip.io`. For
  `<IP>` = `203.0.113.7` that's `203-0-113-7.sslip.io`. [sslip.io](https://sslip.io)
  is a free DNS service that resolves that name to that IP, so Caddy can get
  a real Let's Encrypt certificate for it. Nothing to register.
* **Own domain?** Create a DNS **A record**, e.g. `coach.example.com → <IP>`,
  wait a few minutes until `ping coach.example.com` shows `<IP>`, then set
  `DOMAIN=coach.example.com`.

**Optional settings** (all documented at the bottom of `.env.example`):

```dotenv
DEMO_RATE_LIMIT_PER_MINUTE=3      # questions per visitor per minute
DEMO_RATE_LIMIT_PER_DAY=30        # questions per visitor per day
DEMO_RATE_LIMIT_GLOBAL_PER_DAY=0  # cap for ALL visitors together (0 = none)
DEMO_RESET_ON_START=0             # 1 = fresh demo data at every restart
```

**Optional password for the whole site.** Generate a hash (this downloads the
Caddy image the first time):

```bash
docker run --rm caddy:2-alpine caddy hash-password -p 'choose-a-password'
```

and add it to `.env` **inside single quotes**, because the hash contains `$`
signs:

```dotenv
APP_USER=demo
APP_PASSWORD_HASH='$2a$14$....'
```

Leave `APP_PASSWORD_HASH` unset for an open demo.

`.env` is in `.gitignore` and `.dockerignore`, so it's never committed or
copied into the image.

---

## 11. Build and start

```bash
docker compose up -d --build
```

The first build takes **10–20 minutes** on 2 ARM cores. It installs CPU-only
PyTorch, downloads the two embedding/reranking models, and builds the research
index from `corpus/raw/`. Later builds reuse those cached layers and take a
minute or two.

Watch it come up:

```bash
docker compose ps
```

```bash
docker compose logs -f app
```

You should see `[prepare] DEMO_MODE: seeding synthetic demo data`, then
eventually `[Server] Agent ready.` Press Ctrl+C to stop following the logs.
The containers keep running. `docker compose ps` shows the app as
**healthy** after a minute or two.

Open `https://<DOMAIN>` in your browser. Caddy fetches the certificate on the
first request, which can take 10–30 seconds.

**It doesn't load?**

```bash
docker compose logs caddy | tail -50
```

* `timeout during connect` or `connection refused` in Caddy's ACME errors
  means port 80 or 443 is still blocked. Go back to **6a and 6b**.
* `DOMAIN` doesn't resolve to the VM: check `ping <DOMAIN>`.
* Too many failed attempts can make Let's Encrypt rate-limit you for an hour.
  Fix the cause, then `docker compose restart caddy`.

From the VM itself you can test the app without Caddy:

```bash
docker compose exec app python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:3000/status').read())"
```

---

## 12. Day-2 operations

**Logs**

```bash
docker compose logs -f --tail=200 app
```

```bash
docker compose logs -f --tail=200 caddy
```

**Restart / stop**

```bash
docker compose restart app
```

```bash
docker compose down
```

`down` keeps your data volume. Both containers come back automatically after
a VM reboot (`restart: unless-stopped`).

**Update to a newer commit**

```bash
git pull
```

```bash
docker compose up -d --build
```

```bash
docker image prune -f
```

Your data volume is kept across updates. If an update changes the research
corpus and you want the new index (or simply a clean demo), reset the demo
data as below.

**Reset the demo to fresh synthetic data**

Run this once:

```bash
DEMO_RESET_ON_START=1 docker compose up -d --force-recreate app
```

Then put the app back to normal, so later restarts keep their data:

```bash
docker compose up -d --force-recreate app
```

Alternatively, set `DEMO_RESET_ON_START=1` in `.env` to reset at every restart.

**Back up the data volume**

The volume is named `fitnotes-coach_app-data`. Archive it into the current
directory:

```bash
docker run --rm -v fitnotes-coach_app-data:/data -v "$PWD":/backup alpine tar czf /backup/app-data-$(date +%F).tgz -C /data .
```

Copy the archive to your own computer (run this **on your computer**):

```bash
scp -i ~/.ssh/oracle_fitnotes ubuntu@<IP>:FitNotes-Agentic-AI-Coach/app-data-*.tgz .
```

Restore into the volume (stops the app briefly):

```bash
docker compose stop app
```

```bash
docker run --rm -v fitnotes-coach_app-data:/data -v "$PWD":/backup alpine sh -c "rm -rf /data/* && tar xzf /backup/app-data-YYYY-MM-DD.tgz -C /data"
```

```bash
docker compose start app
```

**Watch your Gemini usage** at <https://aistudio.google.com> → your key →
usage. If the demo is popular, lower `DEMO_RATE_LIMIT_PER_DAY` or set
`DEMO_RATE_LIMIT_GLOBAL_PER_DAY`, then run `docker compose up -d`.

**Disk and memory**

```bash
df -h /
```

```bash
docker stats --no-stream
```

```bash
docker system df
```

---

## 13. Running your own (non-demo) copy

To use the stack for **your own** FitNotes data instead of a public demo, set
`DEMO_MODE=0` in `.env` **and** set `APP_PASSWORD_HASH` (step 10). Without a
password, anyone who finds the URL can read your training history. On first
start the volume has no workout DB. Open the site and upload your
`.fitnotes` backup from the attach (📎) menu.

---

## 14. Running elsewhere

**Any VPS** (Hetzner, DigitalOcean, Lightsail, a home server, …) works the
same way: install Docker, open ports 80/443 in its firewall, then follow steps
8–12. Recommended minimum: **2 vCPU and 4 GB RAM** (2 GB works with swap).
`amd64` and `arm64` are both supported. The Dockerfile installs the CPU build
of PyTorch on either.

**Google Cloud Run** can run the image (point it at port **3000**, no Caddy
needed, since Cloud Run terminates HTTPS), but be aware of what this app
assumes:

* **Ephemeral disk.** `data/` (SQLite DB, Chroma index, memory, chat
  checkpoints) lives on the container's local disk and is lost whenever an
  instance is replaced. With `DEMO_MODE=1` it simply re-seeds, which is
  acceptable for a throwaway demo but not for real data. Don't put the SQLite
  files on a Cloud Storage FUSE mount, because SQLite needs real file locking.
* **Scale to zero means slow cold starts.** The first request after idling
  waits for Python, PyTorch, the models and the agent to start (often 30–90
  s). `--min-instances=1` avoids that, but it isn't free.
* **Exactly one instance.** The shared session, its lock and the rate limiter
  are in memory, so set `--max-instances=1`.
* **CPU must stay allocated** (`--no-cpu-throttling`). The agent initialises in
  the background after start-up, and Cloud Run otherwise throttles CPU
  outside requests.
* Raise the request timeout (`--timeout=900`), because agent turns can take
  minutes.
* The memory limit should be at least 2 GiB.

A small always-on VM (like the Oracle one above) is simpler for this app.
