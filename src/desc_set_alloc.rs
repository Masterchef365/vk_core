use anyhow::Result;
use erupt::vk1_0 as vk;
use crate::SharedCore;

/// Descriptor set allocator, creates pools of a fixed size and creates more as needed
/// This might be a bad idea
pub struct DescriptorSetAllocator {
    template: Vec<vk::DescriptorPoolSizeBuilder<'static>>,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
    pools: Vec<vk::DescriptorPool>,
    current_sets: Vec<vk::DescriptorSet>,
    alloc_size: u32,
    core: SharedCore,
}

impl DescriptorSetAllocator {
    /// Create a new descriptor set allocator, which creates pools of the alloc_size given a
    /// template
    pub fn new(
        core: SharedCore,
        mut template: Vec<vk::DescriptorPoolSizeBuilder<'static>>,
        descriptor_set_layout: vk::DescriptorSetLayout,
        alloc_size: u32,
    ) -> Self {
        for dpsb in &mut template {
            dpsb.descriptor_count *= alloc_size;
        }

        let descriptor_set_layouts = vec![descriptor_set_layout; alloc_size as usize];

        Self {
            template,
            core,
            descriptor_set_layouts,
            current_sets: Vec::new(),
            alloc_size,
            pools: Vec::new(),
        }
    }

    fn allocate_more_sets(&mut self) -> Result<()> {
        // Create descriptor pool of appropriate size
        let create_info = vk::DescriptorPoolCreateInfoBuilder::new()
            .pool_sizes(&self.template)
            .max_sets(self.alloc_size); // TODO: Some ops might not need descriptor sets at all! This is potentially wasteful
        let descriptor_pool = unsafe {
            self.core
                .device
                .create_descriptor_pool(&create_info, None, None)
        }
        .result()?;

        // Create descriptor sets
        let create_info = vk::DescriptorSetAllocateInfoBuilder::new()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&self.descriptor_set_layouts);
        let descriptor_sets =
            unsafe { self.core.device.allocate_descriptor_sets(&create_info) }.result()?;

        self.pools.push(descriptor_pool);
        self.current_sets = descriptor_sets;

        Ok(())
    }

    // My stroop was rather waffle
    /// Get a new descriptor set
    pub fn pop(&mut self) -> Result<vk::DescriptorSet> {
        if let Some(set) = self.current_sets.pop() {
            Ok(set)
        } else {
            self.allocate_more_sets()?;
            Ok(self.current_sets.pop().unwrap())
        }
    }
}

impl Drop for DescriptorSetAllocator {
    fn drop(&mut self) {
        unsafe {
            for pool in self.pools.drain(..) {
                self.core.device.destroy_descriptor_pool(Some(pool), None);
            }
        }
    }
}

