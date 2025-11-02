import adapter from '@sveltejs/adapter-auto';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    kit: {
        // adapter-auto 仅支持某些环境，请参阅 https://kit.svelte.dev/docs/adapter-auto 获取列表。
        // 如果您的环境不支持或配置不受支持，请使用适当的适配器。
        adapter: adapter()
    }
};

export default config;