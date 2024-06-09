import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

const config: Config = {
  title: "MyDatahack",
  tagline: "Imperfection is the fingerprint of your soul...",
  favicon: "img/icon-circle.png",

  // Set the production url of your site here
  url: "https://mydatahack.com",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "mdh", // Usually your GitHub org/user name.
  projectName: "docusaurus-tech-blog-example", // Usually your repo name.

  onBrokenLinks: "throw",
  onBrokenMarkdownLinks: "warn",

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },
  plugins: [
    [
      "@docusaurus/plugin-content-blog",
      {
        id: "web-technologies",
        routeBasePath: "web-technologies",
        path: "./web-technologies",
        showReadingTime: true,
      },
    ],
    [
      "@docusaurus/plugin-content-blog",
      {
        id: "data-engineering",
        routeBasePath: "data-engineering",
        path: "./data-engineering",
        showReadingTime: true,
      },
    ],
    [
      "@docusaurus/plugin-content-blog",
      {
        id: "data-science",
        routeBasePath: "data-science",
        path: "./data-science",
        showReadingTime: true,
        blogSidebarCount: 20,
        blogSidebarTitle: "Posts",
      },
    ],
    [
      "@docusaurus/plugin-content-blog",
      {
        id: "infrastructure",
        routeBasePath: "infrastructure",
        path: "./infrastructure",
        showReadingTime: true,
      },
    ],
  ],
  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: "./sidebars.ts",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: "MyDatahack",
      logo: {
        alt: "MyDatahack Logo",
        src: "img/icon-circle.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Blogs",
        },
        {
          to: "/web-technologies",
          label: "Web Technologies",
          position: "left",
        },
        {
          to: "/data-engineering",
          label: "Data Engineering",
          position: "left",
        },
        { to: "/data-science", label: "Data Science", position: "left" },
        { to: "/infrastructure", label: "Infrastructure", position: "left" },

        {
          href: "https://github.com/takahirohonda",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "More",
          items: [
            {
              label: "Blog",
              to: "/blog",
            },
            {
              label: "GitHub",
              href: "https://github.com/takahirohonda",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} MDH.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
