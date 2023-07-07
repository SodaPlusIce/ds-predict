import {
  HomeOutlined,
  BankOutlined,
  UserOutlined,
  AuditOutlined,
  DashboardOutlined,
  InfoCircleOutlined,
  ApiOutlined
} from '@ant-design/icons'
import Home from '@/pages/home'
import Workspace from '@/pages/home/Workspace'
import ErrorPage from '@/pages/public/errorPage'

import UserList from '@/pages/user/list'
import UserEdit from '@/pages/user/edit'

import RoleList from '@/pages/role/list'

import AuthTest from '@/pages/test'
import { MenuRoute } from '@/route/types'
// import React from 'react'
// import { Icon } from '@iconify/react'
import { TestApiLoad } from './TempTestRouteComponent'

/**
 * path 跳转的路径
 * component 对应路径显示的组件
 * exact 匹配规则，true的时候则精确匹配。
 */
const preDefinedRoutes: MenuRoute[] = [
  {
    path: '/',
    name: '首页',
    exact: true,
    key: 'home',
    icon: HomeOutlined,
    component: Home
  },
]

export default preDefinedRoutes
