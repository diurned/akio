use std::fmt;

pub struct MemInfo {
    pub total: u64,
    pub available: u64,
    pub used: u64,
}

impl MemInfo {
    fn used_percent(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.used as f64 / self.total as f64) * 100.0
    }
}

impl fmt::Display for MemInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Total:     {}\nUsed:      {} ({:.1}%)\nAvailable: {}",
            format_bytes(self.total),
            format_bytes(self.used),
            self.used_percent(),
            format_bytes(self.available),
        )
    }
}

fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    const KIB: u64 = 1024;

    if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// Linux: parse /proc/meminfo
#[cfg(target_os = "linux")]
pub fn get_mem_info() -> Result<MemInfo, String> {
    use std::fs;

    let content = fs::read_to_string("/proc/meminfo")
        .map_err(|e| format!("Failed to read /proc/meminfo: {e}"))?;

    let mut total: Option<u64> = None;
    let mut available: Option<u64> = None;

    for line in content.lines() {
        let mut parts = line.split_whitespace();
        match parts.next() {
            Some("MemTotal:") => {
                total = parts.next().and_then(|v| v.parse::<u64>().ok()).map(|kb| kb * 1024);
            }
            Some("MemAvailable:") => {
                available = parts.next().and_then(|v| v.parse::<u64>().ok()).map(|kb| kb * 1024);
            }
            _ => {}
        }
        if total.is_some() && available.is_some() {
            break;
        }
    }

    let total = total.ok_or("MemTotal not found in /proc/meminfo")?;
    let available = available.ok_or("MemAvailable not found in /proc/meminfo")?;

    Ok(MemInfo {
        total,
        available,
        used: total.saturating_sub(available),
    })
}

// ── macOS: use sysctl + host_statistics
#[cfg(target_os = "macos")]
pub fn get_mem_info() -> Result<MemInfo, String> {
    use std::mem;

    // total RAM via sysctl("hw.memsize")
    unsafe extern "C" {
        fn sysctlbyname(
            name: *const i8,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
    }

    let total: u64 = unsafe {
        let name = b"hw.memsize\0";
        let mut value: u64 = 0;
        let mut size = mem::size_of::<u64>();
        let ret = sysctlbyname(
            name.as_ptr() as *const i8,
            &mut value as *mut u64 as *mut _,
            &mut size,
            std::ptr::null_mut(),
            0,
        );
        if ret != 0 {
            return Err("sysctlbyname(hw.memsize) failed".into());
        }
        value
    };

    // page size via sysconf
    unsafe extern "C" {
        fn sysconf(name: i32) -> i64;
    }
    const _SC_PAGESIZE: i32 = 29;
    let page_size = unsafe { sysconf(_SC_PAGESIZE) as u64 };

    // vm_statistics64 via host_statistics64
    #[repr(C)]
    #[derive(Default)]
    struct VmStatistics64 {
        free_count: u32,
        active_count: u32,
        inactive_count: u32,
        wire_count: u32,
        zero_fill_count: u64,
        reactivations: u64,
        pageins: u64,
        pageouts: u64,
        faults: u64,
        cow_faults: u64,
        lookups: u64,
        hits: u64,
        purges: u64,
        purgeable_count: u32,
        speculative_count: u32,
        decompressions: u64,
        compressions: u64,
        swapins: u64,
        swapouts: u64,
        compressor_page_count: u32,
        throttled_count: u32,
        external_page_count: u32,
        internal_page_count: u32,
        total_uncompressed_pages_in_compressor: u64,
    }

    unsafe extern "C" {
        fn mach_host_self() -> u32;
        fn host_statistics64(
            host: u32,
            flavor: i32,
            host_info: *mut VmStatistics64,
            host_info_count: *mut u32,
        ) -> i32;
    }

    const HOST_VM_INFO64: i32 = 4;
    // size in "natural_t" (u32) units
    let count = (mem::size_of::<VmStatistics64>() / mem::size_of::<u32>()) as u32;

    let available: u64 = unsafe {
        let host = mach_host_self();
        let mut stats = VmStatistics64::default();
        let mut info_count = count;
        let ret = host_statistics64(host, HOST_VM_INFO64, &mut stats, &mut info_count);
        if ret != 0 {
            return Err("host_statistics64 failed".into());
        }
        // "available" ≈ free + inactive (reclaimable) pages
        (stats.free_count as u64 + stats.inactive_count as u64) * page_size
    };

    Ok(MemInfo {
        total,
        available,
        used: total.saturating_sub(available),
    })
}

// Windows: use GlobalMemoryStatusEx
#[cfg(target_os = "windows")]
pub fn get_mem_info() -> Result<MemInfo, String> {
    use std::mem;

    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    unsafe extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        dw_length: mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };

    let ret = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ret == 0 {
        return Err("GlobalMemoryStatusEx failed".into());
    }

    let total = status.ull_total_phys;
    let available = status.ull_avail_phys;

    Ok(MemInfo {
        total,
        available,
        used: total.saturating_sub(available),
    })
}

// FreeBSD / OpenBSD / NetBSD
#[cfg(any(target_os = "freebsd", target_os = "openbsd", target_os = "netbsd"))]
fn get_mem_info() -> Result<MemInfo, String> {
    use std::mem;

    extern "C" {
        fn sysctlbyname(
            name: *const i8,
            oldp: *mut std::ffi::c_void,
            oldlenp: *mut usize,
            newp: *mut std::ffi::c_void,
            newlen: usize,
        ) -> i32;
        fn sysconf(name: i32) -> i64;
    }

    let read_u64 = |name: &[u8]| -> Result<u64, String> {
        unsafe {
            let mut value: u64 = 0;
            let mut size = mem::size_of::<u64>();
            let ret = sysctlbyname(
                name.as_ptr() as *const i8,
                &mut value as *mut u64 as *mut _,
                &mut size,
                std::ptr::null_mut(),
                0,
            );
            if ret == 0 {
                Ok(value)
            } else {
                // fallback: try as u32
                let mut v32: u32 = 0;
                let mut s32 = mem::size_of::<u32>();
                let r2 = sysctlbyname(
                    name.as_ptr() as *const i8,
                    &mut v32 as *mut u32 as *mut _,
                    &mut s32,
                    std::ptr::null_mut(),
                    0,
                );
                if r2 == 0 { Ok(v32 as u64) } else { Err(format!("sysctl failed for {:?}", std::str::from_utf8(name))) }
            }
        }
    };

    const _SC_PAGESIZE: i32 = 47; // FreeBSD value; close enough for NetBSD/OpenBSD
    let page_size = unsafe { sysconf(_SC_PAGESIZE) as u64 };

    let total = read_u64(b"hw.physmem\0")?;
    let free_pages = read_u64(b"vm.stats.vm.v_free_count\0")
        .or_else(|_| read_u64(b"vm.v_free_count\0"))?;
    let inactive_pages = read_u64(b"vm.stats.vm.v_inactive_count\0")
        .or_else(|_| read_u64(b"vm.v_inactive_count\0"))
        .unwrap_or(0);

    let available = (free_pages + inactive_pages) * page_size;

    Ok(MemInfo {
        total,
        available,
        used: total.saturating_sub(available),
    })
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "macos",
    target_os = "windows",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
)))]
pub fn get_mem_info() -> Result<MemInfo, String> {
    Err(format!("Unsupported platform: {}", std::env::consts::OS))
}
