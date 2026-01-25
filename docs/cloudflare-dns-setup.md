# Cloudflare DNS Setup for blackroad.io

## Required DNS Changes

Remove the current A record and add CNAME records pointing to Vercel.

### Step 1: Delete Existing A Record

| Type | Name | Content |
|------|------|---------|
| A | @ | 216.198.79.1 | ← **DELETE THIS** |

### Step 2: Add CNAME Records

| Type | Name | Target | Proxy Status |
|------|------|--------|--------------|
| CNAME | @ | cname.vercel-dns.com | DNS only (grey cloud) |
| CNAME | www | cname.vercel-dns.com | DNS only (grey cloud) |

## Important Notes

- **Proxy must be OFF** (grey cloud, not orange) for Vercel SSL to work
- Changes propagate within 5 minutes typically
- Vercel will automatically provision SSL certificates once DNS is configured

## Verification

After making changes, verify with:

```bash
dig blackroad.io CNAME +short
# Should return: cname.vercel-dns.com

dig www.blackroad.io CNAME +short
# Should return: cname.vercel-dns.com
```

## Cloudflare Dashboard Path

1. https://dash.cloudflare.com
2. Select **blackroad.io** zone
3. Click **DNS** in sidebar
4. Make the changes listed above
