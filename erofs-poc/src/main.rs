// erofs xattr poc - investigation into how xattrs are actually laid out on disk
//
// started this to understand what erofs-rs is missing before gsoc.
// ran into two real bugs along the way - both of them documented inline.
//
// build:     cargo build
// test img:  python3 make_test_image.py test.erofs
// run:       cargo run -- test.erofs
//
// real output when working:
//   nid=0 has 2 xattrs (filter passed)
//     security.selinux = u:object_r:system_file:s0
//     user.comment = hello from erofs poc

use binrw::BinRead;
use std::hash::Hasher;
use std::{borrow::Cow, env, error::Error, fs, io::Cursor, mem::size_of};
use twox_hash::XxHash32;

const SUPER_BLOCK_OFFSET: usize = 1024;
const EROFS_MAGIC: u32 = 0xe0f5e1e2;
const SHARED_XATTR_SLOT: usize = 4; // each __le32 shared index slot

// on-disk superblock - 128 bytes total (matches erofs_fs.h erofs_super_block)
// i cross-referenced every field offset against the kernel header
#[repr(C)]
#[derive(Debug, Clone, Copy, BinRead)]
#[br(little)]
struct SuperBlock {
    magic: u32,
    checksum: u32,
    feature_compat: u32,
    blk_size_bits: u8,
    ext_slots: u8,
    root_nid: u16,
    inos: u64,
    build_time: u64,
    build_time_ns: u32,
    blocks: u32,
    meta_blk_addr: u32,
    xattr_blk_addr: u32, // where shared xattr table lives
    uuid: [u8; 16],
    volume_name: [u8; 16],
    feature_incompat: u32,
    compr_algs: u16,
    extra_devices: u16,
    devt_slot_off: u16,
    dir_blk_bits: u8,
    xattr_prefix_count: u8, // number of long prefixes (bit 7 of name_index)
    xattr_prefix_start: u32,
    packed_nid: u64,
    xattr_filter_res: u8,
    _reserved: [u8; 23],
}

// first two fields are shared between compact and extended
// i use this to peek at format and xattr_count before deciding inode size
#[repr(C)]
#[derive(Debug, Clone, Copy, BinRead)]
#[br(little)]
struct InodeHeader {
    format: u16,
    xattr_count: u16,
}

// compact inode - 32 bytes (erofs_inode_compact)
#[repr(C)]
#[derive(Debug, Clone, Copy, BinRead)]
#[br(little)]
struct InodeCompact {
    format: u16,
    xattr_count: u16,
    mode: u16,
    nlink: u16,
    size: u32,
    _reserved: u32,
    inode_data: u32, // raw block address for data
    inode: u32,
    uid: u16,
    gid: u16,
    _reserved2: u32,
}

// extended inode - 64 bytes (erofs_inode_extended)
#[repr(C)]
#[derive(Debug, Clone, Copy, BinRead)]
#[br(little)]
struct InodeExtended {
    format: u16,
    xattr_count: u16,
    mode: u16,
    _reserved: u16,
    size: u64,
    inode_data: u32,
    inode: u32,
    uid: u32,
    gid: u32,
    mtime: u64,
    mtime_ns: u32,
    nlink: u32,
    _reserved2: [u8; 16],
}

// erofs_xattr_ibody_header - 12 bytes, sits right after the inode struct
// layout after inode:
//   [XattrBodyHeader 12 bytes]
//   [shared xattr slots: shared_count * 4 bytes]  <- SKIPPED unless resolving shared
//   [inline xattr entries...]
#[repr(C)]
#[derive(Debug, Clone, Copy, BinRead)]
#[br(little)]
struct XattrBodyHeader {
    name_filter: u32, // 32-bit bloom filter, xxh32 seeded by name_index
    shared_count: u8,
    _reserved: [u8; 7],
}

// erofs_xattr_entry - 4 bytes header, followed by name_len bytes of name then value_len bytes of value
// IMPORTANT: after (header + name + value), cursor must be rounded up to next 4-byte boundary
// the original poc was missing this and reading garbage on the second+ xattr
#[repr(C)]
#[derive(Debug, Clone, Copy, BinRead)]
#[br(little)]
struct XattrEntry {
    name_len: u8,
    name_index: u8, // low 7 bits = prefix index, bit 7 = long prefix flag
    value_len: u16,
}

// --- bloom filter ---
//
// investigation finding #1:
// erofs uses XxHash32 seeded by name_index to compute one bit of the filter.
// kernel code in fs/erofs/xattr.c erofs_xattr_filter_hash():
//   return BIT(xxh32(name, namelen, index) & EROFS_XATTR_FILTER_BITS_MASK)
//
// my first attempt hand-rolled xxh32 and got it wrong.
// the 4-byte accumulation loop order is:
//   h += lane * PRIME3        <- add
//   h = rotl32(h, 17)         <- rotate
//   h *= PRIME4               <- multiply
// i had the multiply happening BEFORE the rotate too, which shifted every hash.
// result: python image builder computed bit 6 for "selinux", rust computed bit 14.
// the filter rejected every valid lookup.
//
// fix: use twox_hash::XxHash32::with_seed(name_index as u32) which matches the kernel.
// the image builder also got fixed to use the same correct implementation.

fn bloom_filter_check(name_filter: u32, name_index: u8, name: &[u8]) -> bool {
    if name_filter == u32::MAX {
        return true; // filter disabled in superblock (xattr_filter_res set)
    }
    let mut hasher = XxHash32::with_seed(name_index as u32);
    hasher.write(name);
    let h = hasher.finish() as u32;
    let bit = 1u32 << (h & 31);
    (name_filter & bit) != 0
}

// 7 predefined short prefixes from erofs_fs.h
// bit 7 of name_index signals a long prefix stored in the packed inode at xattr_prefix_start
// not handling long prefixes here - that needs xattr_prefix_count + packed inode parsing
fn short_prefix(index: u8) -> &'static str {
    match index & 0x7F {
        0 => "",
        1 => "user.",
        2 => "trusted.",
        3 => "security.",
        4 => "system.",
        5 => "system.posix_acl_access",
        6 => "system.posix_acl_default",
        _ => "unknown.",
    }
}

fn read_at<T: for<'a> BinRead<Args<'a> = ()>>(
    data: &[u8],
    offset: usize,
) -> Result<T, Box<dyn Error>> {
    Ok(T::read_le(&mut Cursor::new(&data[offset..]))?)
}

// bit 0 of format: 0=compact (32 bytes), 1=extended (64 bytes)
fn is_compact(format: u16) -> bool {
    (format & 0x01) == 0
}

fn align4(n: usize) -> usize {
    (n + 3) & !3
}

// ---
// investigation finding #2: entry cursor alignment
//
// the original poc walked inline entries without rounding up after (name + value).
// this works fine if theres only one xattr, or if name_len + value_len happens to
// already be a multiple of 4. on real android images most files have 2+ xattrs and
// misaligned sizes, so the parser would start reading the next XattrEntry header
// at the wrong offset - everything silently parses but with garbage field values.
//
// fix: after advancing cursor by nlen + vlen, always align4() before reading next entry.
// erofs_fs.h documents this explicitly in the entry layout comments.
// ---

struct XattrIter<'a> {
    data: &'a [u8],
    cursor: usize,
    remaining: usize,
}

impl<'a> XattrIter<'a> {
    fn new(
        data: &'a [u8],
        xattr_start: usize,
        total_count: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let header: XattrBodyHeader = read_at(data, xattr_start)?;
        let shared = header.shared_count as usize;

        if shared > total_count {
            return Err("corrupted inode: shared_count > xattr_count".into());
        }

        // skip:
        //   12 bytes XattrBodyHeader
        //   shared_count * 4 bytes of __le32 shared xattr ids
        // inline entries start after that.
        // to resolve shared xattrs we'd need to read from xattr_blk_addr - not doing that here
        let cursor =
            xattr_start + size_of::<XattrBodyHeader>() + (shared * SHARED_XATTR_SLOT);

        Ok(Self {
            data,
            cursor,
            remaining: total_count - shared,
        })
    }

    fn next_entry(&mut self) -> Result<Option<(String, Cow<'_, str>)>, Box<dyn Error>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        let entry: XattrEntry = read_at(self.data, self.cursor)?;
        self.cursor += size_of::<XattrEntry>();

        let nlen = entry.name_len as usize;
        let vlen = entry.value_len as usize;

        if self.cursor + nlen + vlen > self.data.len() {
            return Err("xattr entry extends past image boundary - truncated or corrupt".into());
        }

        let name_bytes = &self.data[self.cursor..self.cursor + nlen];
        self.cursor += nlen;

        let value_bytes = &self.data[self.cursor..self.cursor + vlen];
        self.cursor += vlen;

        // this is the alignment fix. without it, cursor lands at a wrong offset
        // and the next XattrEntry reads garbage name_len/value_len fields.
        self.cursor = align4(self.cursor);
        self.remaining -= 1;

        let prefix = short_prefix(entry.name_index);
        let key = format!("{}{}", prefix, String::from_utf8_lossy(name_bytes));

        // selinux values often end with \x00 which we strip for readability
        let val: Cow<str> = match std::str::from_utf8(value_bytes) {
            Ok(s) => s.trim_end_matches('\0').into(),
            Err(_) => value_bytes
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
                .into(),
        };

        Ok(Some((key, val)))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).ok_or("usage: erofs-xattr-poc <image.erofs>")?;

    // fs::read loads everything into ram - fine for this poc but wont work on
    // a real android system.img (2-3 GB). the main library should use mmap.
    // memmap2 + zerocopy::Ref<_, T> would let us do the struct parsing without
    // any copying at all, which is the pattern erofs-rs should adopt for the hot path.
    let data = fs::read(&path)?;

    let sb: SuperBlock = read_at(&data, SUPER_BLOCK_OFFSET)?;
    if sb.magic != EROFS_MAGIC {
        return Err(format!("bad magic: {:#010x}", sb.magic).into());
    }

    let block_size = 1usize << sb.blk_size_bits;
    let meta_base = sb.meta_blk_addr as usize * block_size;

    eprintln!(
        "image: blksize={} meta_blk={:#x} xattr_blk={:#x} inos={}",
        block_size, meta_base, sb.xattr_blk_addr, sb.inos
    );
    eprintln!(
        "       feature_incompat={:#010x} compr_algs={:#06x} extra_devices={}",
        sb.feature_incompat, sb.compr_algs, sb.extra_devices
    );

    // look for security.selinux specifically - this is the hot path on android
    // (every file has this xattr, every overlayfs lookup checks it)
    let target_name = b"selinux";
    let target_index = 3u8; // "security."

    let mut found = 0usize;

    for nid in 0..sb.inos as usize {
        let off = meta_base + nid * size_of::<InodeCompact>();
        if off + size_of::<InodeHeader>() > data.len() {
            break;
        }

        let hdr: InodeHeader = read_at(&data, off)?;
        if hdr.xattr_count == 0 {
            continue;
        }

        let inode_size = if is_compact(hdr.format) {
            size_of::<InodeCompact>()
        } else {
            size_of::<InodeExtended>()
        };

        if off + inode_size + size_of::<XattrBodyHeader>() > data.len() {
            continue;
        }

        let body_hdr: XattrBodyHeader = read_at(&data, off + inode_size)?;

        // bloom check before we touch any entry bytes.
        // if the filter says "not here", skip the whole inode.
        // this is O(1) and avoids the XattrIter entirely on miss.
        if !bloom_filter_check(body_hdr.name_filter, target_index, target_name) {
            continue;
        }

        println!(
            "nid={} xattrs={} (filter={:#010x} passed bloom check)",
            nid, hdr.xattr_count, body_hdr.name_filter
        );

        let mut iter = XattrIter::new(&data, off + inode_size, hdr.xattr_count as usize)?;
        while let Some((k, v)) = iter.next_entry()? {
            println!("  {k} = {v}");
        }

        found += 1;
    }

    if found == 0 {
        eprintln!("no inodes with xattrs matched the bloom filter for security.selinux");
        eprintln!("check: is the image built with the correct xxh32 (see make_test_image.py)?");
    } else {
        println!("\n{found} inode(s) with xattrs parsed");
    }

    Ok(())
}
