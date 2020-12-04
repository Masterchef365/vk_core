use erupt::{
    utils::loading::DefaultEntryLoader, vk1_0 as vk, DeviceLoader, EntryLoader, InstanceLoader, cstr
};
use gpu_alloc::{self, GpuAllocator};
use std::sync::{Arc, Mutex, MutexGuard};
use anyhow::{format_err, Result};

/// The core of a simple vulkan application; meant to be shared between several objects.
pub struct Core {
    pub allocator: Mutex<GpuAllocator<vk::DeviceMemory>>,
    pub queue: vk::Queue,
    pub device: DeviceLoader,
    pub instance: InstanceLoader,
    pub _entry: DefaultEntryLoader,
}

/// All structures outside of the core generated while the core is made
pub struct CoreMeta {
    pub queue_family_index: u32,
    pub physical_device: vk::PhysicalDevice,
}

/// Shareable Core, cloning is cheap.
pub type SharedCore = Arc<Core>;

impl Core {
    /// Create a Core suitable for simple GPGPU
    pub fn compute(validation: bool, name: &str) -> Result<(SharedCore, CoreMeta)> {
        let entry = EntryLoader::new()?;

        // Instance
        let name = std::ffi::CString::new(name)?;
        let app_info = vk::ApplicationInfoBuilder::new()
            .application_name(&name)
            .application_version(vk::make_version(1, 0, 0))
            .engine_name(&name)
            .engine_version(vk::make_version(1, 0, 0))
            .api_version(vk::make_version(1, 0, 0));

        // Instance and device layers and extensions
        let mut instance_layers = Vec::new();
        let mut instance_extensions = Vec::new();
        let mut device_layers = Vec::new();
        let device_extensions = Vec::new();

        // Vulkan layers and extensions
        if validation {
            const LAYER_KHRONOS_VALIDATION: *const i8 = cstr!("VK_LAYER_KHRONOS_validation");
            instance_extensions
                .push(erupt::extensions::ext_debug_utils::EXT_DEBUG_UTILS_EXTENSION_NAME);
            instance_layers.push(LAYER_KHRONOS_VALIDATION);
            device_layers.push(LAYER_KHRONOS_VALIDATION);
        }

        // Instance creation
        let create_info = vk::InstanceCreateInfoBuilder::new()
            .application_info(&app_info)
            .enabled_extension_names(&instance_extensions)
            .enabled_layer_names(&instance_layers);

        let instance = InstanceLoader::new(&entry, &create_info, None)?;

        // Hardware selection
        let (queue_family_index, physical_device) = select_compute_device(&instance)?;

        // Create logical device and queues
        let create_info = [vk::DeviceQueueCreateInfoBuilder::new()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0])];

        let physical_device_features = vk::PhysicalDeviceFeaturesBuilder::new();
        let create_info = vk::DeviceCreateInfoBuilder::new()
            .queue_create_infos(&create_info)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&device_extensions)
            .enabled_layer_names(&device_layers);

        let device = DeviceLoader::new(&instance, physical_device, &create_info, None)?;
        let queue = unsafe { device.get_device_queue(queue_family_index, 0, None) };

        // GpuAllocator
        let device_props = unsafe { gpu_alloc_erupt::device_properties(&instance, physical_device)? };
        let allocator =
            GpuAllocator::new(gpu_alloc::Config::i_am_prototyping(), device_props);

        let core = SharedCore::new(Self {
            allocator: Mutex::new(allocator),
            queue,
            device,
            instance,
            _entry: entry,
        });

        let meta = CoreMeta {
            queue_family_index,
            physical_device,
        };

        Ok((core, meta))
    }

    pub fn allocator(&self) -> Result<MutexGuard<GpuAllocator<vk::DeviceMemory>>> {
        self.allocator
            .lock()
            .map_err(|_| format_err!("GpuAllocator mutex poisoned"))
    }
}

fn select_compute_device(instance: &InstanceLoader) -> Result<(u32, vk::PhysicalDevice)> {
    let physical_devices = unsafe { instance.enumerate_physical_devices(None) }.result()?;
    for device in physical_devices {
        let families =
            unsafe { instance.get_physical_device_queue_family_properties(device, None) };
        for (family, properites) in families.iter().enumerate() {
            if properites.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                return Ok((family as u32, device));
            }
        }
    }
    Err(format_err!("No suitable device found"))
}

impl Drop for Core {
    fn drop(&mut self) {
        unsafe { 
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        };
    }
}
